import pickle
import matplotlib

import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from shap import KernelExplainer
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from helpers import get_file_path, load_algerian
from distributions import MultiGaussian, GaussianCopula
from force_dependent import force_dependent_plot
from kernel_dependent import DependentKernelExplainer
import os

save_folder = 'checkpoints/'
figure_folder = 'figures/'

if not os.path.exists(get_file_path(save_folder)):
    os.makedirs(get_file_path(save_folder))

if not os.path.exists(get_file_path(figure_folder)):
    os.makedirs(get_file_path(figure_folder))

seed = 5

np.random.seed(seed)


### Simulated example from introduction

# In this case we can compute the exact values
shap_values = np.array([0.4, 0.1, 0.4, 0])
fig = force_dependent_plot(0.5, shap_values, features=[True, True],
                           feature_names=['prior recidivism/\nhigh heart rate', 'race/\nobesity'], show=False)
fig.savefig(figure_folder + "sim.png", bbox_inches="tight")  # Figure 2 in paper


### Boston Housing dataset

def setup_data():
    x, y = load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    with open(save_folder + 'x_train.txt', 'wb') as fp:
        pickle.dump(x_train, fp)
    with open(save_folder + 'y_train.txt', 'wb') as fp:
        pickle.dump(y_train, fp)
    with open(save_folder + 'x_test.txt', 'wb') as fp:
        pickle.dump(x_test, fp)
    with open(save_folder + 'y_test.txt', 'wb') as fp:
        pickle.dump(y_test, fp)


def setup_forest(save_name):
    with open(save_folder + 'x_train.txt', 'rb') as fp:
        x_train = pickle.load(fp)
    with open(save_folder + 'y_train.txt', 'rb') as fp:
        y_train = pickle.load(fp)
    with open(save_folder + 'x_test.txt', 'rb') as fp:
        x_test = pickle.load(fp)
    with open(save_folder + 'y_test.txt', 'rb') as fp:
        y_test = pickle.load(fp)

    model = xgb.XGBRegressor(n_estimators=1000)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='rmse', early_stopping_rounds=50)

    model.save_model(save_folder + save_name + '.model')
    return model


def setup_linear(save_name):
    with open(save_folder + 'x_train.txt', 'rb') as fp:
        x_train = pickle.load(fp)
    with open(save_folder + 'y_train.txt', 'rb') as fp:
        y_train = pickle.load(fp)

    model = LinearRegression()
    model.fit(x_train, y_train)

    pickle.dump(model, open(save_folder + save_name + '.model', 'wb'))
    return model


def setup_shap(model, save_name, nb_samples=100, x=None, dist=None):
    if x is None:
        with open(save_folder + 'x_train.txt', 'rb') as fp:
            x_train = pickle.load(fp)
    else:
        x_train = x
    if dist is None:
        dist = MultiGaussian(x_train)
    intexp = KernelExplainer(model.predict, x_train)
    condexp = DependentKernelExplainer(model.predict, x_train, dist.sample)

    vals_int = intexp.shap_values(x_train[:nb_samples])
    vals_cond_all = condexp.shap_values(x_train[:nb_samples])

    with open(save_folder + save_name + '_int.txt', 'wb') as fp:
        pickle.dump(vals_int, fp)
    with open(save_folder + save_name + '_cond_all.txt', 'wb') as fp:
        pickle.dump(vals_cond_all, fp)


# Setup everything time-consuming beforehand, this can be commented out when run the second time
# setup_data()
# forest = setup_forest('forest')
# linear = setup_linear('linear')
# setup_shap(forest, 'forest_shaps')
# setup_shap(linear, 'linear_shaps')

## Experiment 1: plot some SHAP values for Boston dataset

with open(save_folder + 'x_train.txt', 'rb') as fp:
    x_train = pickle.load(fp)
with open(save_folder + 'forest_shaps_cond_all.txt', 'rb') as fp:
    forest_shaps_cond_all = pickle.load(fp)

forest = xgb.XGBRegressor(n_estimators=1000)
forest.load_model(save_folder + 'forest.model')

data = load_boston()
sample_index = 16
plt.close()
fig = force_dependent_plot(np.mean(forest.predict(x_train)), forest_shaps_cond_all[sample_index], features=x_train[sample_index],
                           feature_names=data['feature_names'], show=False, text_rotation=-90)
fig.savefig(figure_folder + "boston{}.png".format(sample_index), bbox_inches="tight")  # Figure 3 in paper
plt.close()


## Experiment 2: interventional SHAP part for model output correction

def compute_model_output_correction(all_vals, model, max_removed, X=None, dist=None):
    "Returns model output when most negative features are iterativly mean imputed."
    nb_samples, nb_inputs = all_vals[0].shape
    if X is None:
        with open(save_folder + 'x_train.txt', 'rb') as fp:
            x_train = pickle.load(fp)
    else:
        x_train = X
    if dist is None:
        means = np.tile(np.average(x_train, axis=0), (nb_samples, 1))
    else:
        means = np.full((nb_samples, nb_inputs), np.nan)
    all_fs = [np.empty((nb_samples, max_removed + 1)) for _ in range(len(all_vals))]
    for fs in all_fs:
        fs[:, 0] = model.predict(x_train[:nb_samples])
    temp = x_train[:nb_samples].copy()

    for i in range(max_removed):
        for fs, vals in zip(all_fs, all_vals):
            ix = np.argmin(vals, axis=1)
            mask = np.zeros(vals.shape, bool)
            mask[range(0, nb_samples), ix] = 1
            vals[(vals < 0) & mask] = np.inf
            if dist is not None:
                for j, row in enumerate(vals):
                    if vals[j, ix[j]] == np.inf:  # only compute if it has to be replaced
                        cond_features = (row != np.inf).nonzero()[0]
                        uncond_feature_means = \
                            dist.sample(cond_features, temp[j][cond_features], 1, return_moments=True)[1]
                        means[j, ix[j]] = uncond_feature_means[np.where((row == np.inf).nonzero()[0] == ix[j])]
            temp[vals == np.inf] = means[vals == np.inf]
            assert not (means[vals == np.inf] == np.nan).any()
            fs[:, i + 1] = model.predict(temp)
            temp[:] = x_train[:nb_samples]

    return all_fs


def plot_simple(vals, color, label=None):
    mean = np.average(vals, axis=0)
    plt.plot(range(0, len(mean)), mean, color=color, label=label)


def plot_with_std(vals, color, label=None):
    mean = np.average(vals, axis=0)
    plt.plot(range(0, len(mean)), mean, color=color, label=label)
    std = np.std(vals, axis=0)
    plt.fill_between(range(0, len(mean)), mean + std, mean - std, facecolor=color, alpha=0.1)


def plot_with_faded_trajectories(vals, color, label=None):
    mean = np.average(vals, axis=0)
    plt.plot(range(0, len(mean)), mean, color=color, label=label)
    for row in vals[np.random.choice(len(vals), 10)]:
        plt.plot(range(0, len(mean)), row, color=color, alpha=0.1)


# Linear case

linear = pickle.load(open(save_folder + 'linear.model', 'rb'))
with open(save_folder + 'linear_shaps_int.txt', 'rb') as fp:
    linear_shaps_int = pickle.load(fp)
with open(save_folder + 'linear_shaps_cond_all.txt', 'rb') as fp:
    linear_shaps_cond_all = pickle.load(fp)

nb_features = linear_shaps_int.shape[1]

linear_int_fs, linear_cond_fs, linear_int_part_fs = compute_model_output_correction(
    [linear_shaps_int, linear_shaps_cond_all[:, :nb_features], linear_shaps_cond_all[:, nb_features:]], linear, 10)

for vals in [linear_int_fs, linear_cond_fs, linear_int_part_fs]:
    vals -= vals[:, 0, np.newaxis]

plt.close()

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

plot_simple(linear_int_fs, 'blue', 'Interventional SHAP')
plot_simple(linear_int_part_fs, 'red', 'Interventional SHAP part')
plot_simple(linear_cond_fs, 'green', 'Conditional SHAP')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Change in median house price (in 1000$)')
plt.savefig(figure_folder + 'linear_diff.png', dpi=400)  # Figure 4 in paper

plt.close()

plot_with_std(linear_int_fs, 'blue', 'Interventional SHAP')
plot_with_std(linear_int_part_fs, 'red', 'Interventional SHAP part')
plot_with_std(linear_cond_fs, 'green', 'Conditional SHAP')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Change in median house price (in 1000$)')
plt.savefig(figure_folder + 'linear_diff_std.png', dpi=400)  # Figure 15 in paper

plt.close()

linear_int_fs -= linear_cond_fs
linear_int_part_fs -= linear_cond_fs
plot_with_faded_trajectories(linear_int_fs, 'blue', 'Interventional SHAP')
plot_with_faded_trajectories(linear_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="upper left")
plt.xlabel('Features changed')
plt.ylabel('Difference with conditional SHAP approach (in 1000$)')
plt.savefig(figure_folder + 'linear_norm.png', dpi=400)  # Figure 5 in paper

plt.close()

# Now with conditional means

linear = pickle.load(open(save_folder + 'linear.model', 'rb'))
with open(save_folder + 'linear_shaps_int.txt', 'rb') as fp:
    linear_shaps_int = pickle.load(fp)
with open(save_folder + 'linear_shaps_cond_all.txt', 'rb') as fp:
    linear_shaps_cond_all = pickle.load(fp)

dist_boston = MultiGaussian(x_train)

linear_int_fs, linear_cond_fs, linear_int_part_fs = compute_model_output_correction(
    [linear_shaps_int, linear_shaps_cond_all[:, :nb_features], linear_shaps_cond_all[:, nb_features:]], linear, 10,
    dist=dist_boston)

for vals in [linear_int_fs, linear_cond_fs, linear_int_part_fs]:
    vals -= vals[:, 0, np.newaxis]

plot_simple(linear_int_fs, 'blue', 'Interventional SHAP')
plot_simple(linear_int_part_fs, 'red', 'Interventional SHAP part')
plot_simple(linear_cond_fs, 'green', 'Conditional SHAP')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Change in median house price (in 1000$)')
plt.savefig(figure_folder + 'linear_diff_c.png', dpi=400)  # Figure 6 in paper

plt.close()

linear_int_fs -= linear_cond_fs
linear_int_part_fs -= linear_cond_fs
plot_with_faded_trajectories(linear_int_fs, 'blue', 'Interventional SHAP')
plot_with_faded_trajectories(linear_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="upper left")
plt.xlabel('Features changed')
plt.ylabel('Difference with conditional SHAP approach (in 1000$)')
plt.savefig(figure_folder + 'linear_norm_c.png', dpi=400)  # Figure 7 in paper

plt.close()

# Forest case

forest = xgb.XGBRegressor(n_estimators=1000)
forest.load_model(save_folder + 'forest.model')
with open(save_folder + 'forest_shaps_int.txt', 'rb') as fp:
    forest_shaps_int = pickle.load(fp)
with open(save_folder + 'forest_shaps_cond_all.txt', 'rb') as fp:
    forest_shaps_cond_all = pickle.load(fp)

nb_features = forest_shaps_int.shape[1]
forest_int_fs, forest_cond_fs, forest_int_part_fs = compute_model_output_correction(
    [forest_shaps_int, forest_shaps_cond_all[:, :nb_features], forest_shaps_cond_all[:, nb_features:]], forest, 10)

for vals in [forest_int_fs, forest_cond_fs, forest_int_part_fs]:
    vals -= vals[:, 0, np.newaxis]



plot_simple(forest_int_fs, 'blue', 'Interventional SHAP')
plot_simple(forest_cond_fs, 'green', 'Conditional SHAP')
plot_simple(forest_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Change in median house price (in 1000$)')
plt.savefig(figure_folder + 'forest_diff.png', dpi=400)  # Figure 11 in paper

plt.close()

forest_int_fs -= forest_cond_fs
forest_int_part_fs -= forest_cond_fs
plot_with_faded_trajectories(forest_int_fs, 'blue', 'Interventional SHAP')
plot_with_faded_trajectories(forest_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="upper left")
plt.xlabel('Features changed')
plt.ylabel('Difference with conditional SHAP approach (in 1000$)')
plt.savefig(figure_folder + 'forest_norm.png', dpi=400)  # Figure 12 in paper

plt.close()

# Now with conditional means

forest = xgb.XGBRegressor(n_estimators=1000)
forest.load_model(save_folder + 'forest.model')
with open(save_folder + 'forest_shaps_int.txt', 'rb') as fp:
    forest_shaps_int = pickle.load(fp)
with open(save_folder + 'forest_shaps_cond_all.txt', 'rb') as fp:
    forest_shaps_cond_all = pickle.load(fp)

dist_boston = MultiGaussian(x_train)

forest_int_fs, forest_cond_fs, forest_int_part_fs = compute_model_output_correction(
    [forest_shaps_int, forest_shaps_cond_all[:, :nb_features], forest_shaps_cond_all[:, nb_features:]], forest, 10,
    dist=dist_boston)

for vals in [forest_int_fs, forest_cond_fs, forest_int_part_fs]:
    vals -= vals[:, 0, np.newaxis]

plot_simple(forest_int_fs, 'blue', 'Interventional SHAP')
plot_simple(forest_cond_fs, 'green', 'Conditional SHAP')
plot_simple(forest_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Change in median house price (in 1000$)')
plt.savefig(figure_folder + 'forest_diff_c.png', dpi=400)  # Figure 13 in paper

plt.close()

forest_int_fs -= forest_cond_fs
forest_int_part_fs -= forest_cond_fs
plot_with_faded_trajectories(forest_int_fs, 'blue', 'Interventional SHAP')
plot_with_faded_trajectories(forest_int_part_fs, 'red', 'Interventional SHAP part')
plt.legend(loc="lower right")
plt.xlabel('Features changed')
plt.ylabel('Difference with conditional SHAP approach (in 1000$)')
plt.savefig(figure_folder + 'forest_norm_c.png', dpi=400)  # Figure 14 in paper

plt.close()

### Algerian Forest fires dataset

matplotlib.rcdefaults()

x, y = load_algerian(return_X_y=True)
data = load_algerian()

seed = 11

np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = xgb.XGBClassifier(n_estimators=1000)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

dist = GaussianCopula(x)

condexp = DependentKernelExplainer(lambda d: model.predict(d, output_margin=True), x, dist.sample)

# vals_cond_all = condexp.shap_values(x_test)

# with open(save_folder + 'algerian' + '_cond_all.txt', 'wb') as fp:
#     pickle.dump(vals_cond_all, fp)

with open(save_folder + 'algerian' + '_cond_all.txt', 'rb') as fp:
    vals_cond_all = pickle.load(fp)

for sample_index in [6, 27, 32]:
    fig = force_dependent_plot(np.mean(model.predict(x, output_margin=True)), vals_cond_all[sample_index],
                               features=x_test[sample_index],
                               feature_names=list(data.columns)[:-1], show=False, link='logit')
    fig.savefig(figure_folder + "algerian{}.png".format(sample_index), bbox_inches="tight")  # Figure 8, 10, 9 in paper
