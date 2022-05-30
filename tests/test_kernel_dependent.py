import numpy as np
from shap import KernelExplainer

from distributions import MultiGaussian
from helpers import int_shap_part_exact
from kernel_dependent import DependentKernelExplainer


def test_conditional_shap_toy():
    def toy_model(data):
        return data[:, 0]

    def toy_dist(features, feature_values, n):
        if len(features) == 2:
            print('All features have values!')
            return np.tile(feature_values, (n, 1))
        elif len(features) == 1:
            out = np.full((n, 2), feature_values[0])
            other_feature = int(np.logical_not(features[0]))
            print(other_feature)
            if feature_values[0] == 1:
                out[:, other_feature] = np.random.binomial(1, 0.7, size=n)
            else:
                out[:, other_feature] = np.logical_not(np.random.binomial(1, 0.7, size=n))
            return out
        elif len(features) == 0:
            out = np.zeros((n, 2))
            out[:, 0] = np.random.binomial(1, 0.5, size=n)
            out[:, 1] = np.logical_not(np.logical_xor(np.random.binomial(1, 0.7, size=n), out[:, 0]))
            return out
        else:
            raise Exception

    data = toy_dist([], [], 10000)
    exp = DependentKernelExplainer(toy_model, data, toy_dist)
    vals = exp.shap_values(np.array([[1, 1]]))[0][:2]
    assert np.allclose(vals, [0.4, 0.1], rtol=0, atol=0.01)


def test_interventional_part_toy():
    def toy_model(data):
        return data[:, 0]

    def toy_dist(features, feature_values, n):
        if len(features) == 2:
            print('All features have values!')
            return np.tile(feature_values, (n, 1))
        elif len(features) == 1:
            out = np.full((n, 2), feature_values[0])
            other_feature = int(np.logical_not(features[0]))
            print(other_feature)
            if feature_values[0] == 1:
                out[:, other_feature] = np.random.binomial(1, 0.7, size=n)
            else:
                out[:, other_feature] = np.logical_not(np.random.binomial(1, 0.7, size=n))
            return out
        elif len(features) == 0:
            out = np.zeros((n, 2))
            out[:, 0] = np.random.binomial(1, 0.5, size=n)
            out[:, 1] = np.logical_not(np.logical_xor(np.random.binomial(1, 0.7, size=n), out[:, 0]))
            return out
        else:
            raise Exception

    data = toy_dist([], [], 10000)
    exp = DependentKernelExplainer(toy_model, data, toy_dist)
    vals = exp.shap_values(np.array([[1, 1]]))[0][2:]
    assert np.allclose(vals, [0.4, 0.0], rtol=0, atol=0.05)


def test_independent_dist():
    nb_features = 4

    class SimModel:
        def __init__(self):
            self.coefs = np.random.uniform(-5, 5, size=nb_features)

        def predict(self, x):
            assert x.shape[1] == len(self.coefs)
            return np.inner(x, self.coefs)

    sim_model = SimModel()
    mean = np.random.uniform(-3, 3, size=nb_features)
    cov = np.eye(nb_features)
    sim_data = np.random.multivariate_normal(mean, cov, size=2000)
    dist = MultiGaussian(mean=mean, cov=cov)
    intexp = KernelExplainer(sim_model.predict, sim_data)
    condexp = DependentKernelExplainer(sim_model.predict, sim_data, dist.sample)

    vals_int = intexp.shap_values(sim_data[0])
    vals_cond_all = condexp.shap_values(sim_data[0])

    vals_exact = sim_model.coefs * (sim_data[0] - mean)

    print(np.average(np.absolute(vals_int - vals_exact)))
    assert np.allclose(vals_int, vals_exact, rtol=0, atol=0.2)
    print(np.average(np.absolute(vals_cond_all[:nb_features] - vals_exact)))
    assert np.allclose(vals_cond_all[:nb_features], vals_exact, rtol=0, atol=0.2)
    print(np.average(np.absolute(vals_cond_all[nb_features:] - vals_exact)))
    assert np.allclose(vals_exact, vals_cond_all[nb_features:], rtol=0, atol=0.2)

    # Test arguments
    condexp.shap_values(sim_data[0], neval_kernel=334, nperm_sampling=334)
    assert True


def test_against_all_perms():
    nb_features = 4

    class SimModel:
        def __init__(self):
            self.coefs = np.random.uniform(-5, 5, size=nb_features)

        def predict(self, x):
            assert x.shape[1] == len(self.coefs)
            return np.inner(x, self.coefs)

    sim_model = SimModel()
    mean = np.random.uniform(-3, 3, size=nb_features)
    A = np.random.rand(nb_features, nb_features)
    cov = np.dot(A, A.transpose())
    sim_data = np.random.multivariate_normal(mean, cov, size=2000)
    dist = MultiGaussian(mean=mean, cov=cov)

    condexp = DependentKernelExplainer(sim_model.predict, sim_data, dist.sample)

    vals_int_part = condexp.shap_values(sim_data[0])[nb_features:]
    vals_int_part_exact = int_shap_part_exact(sim_data[0, np.newaxis], sim_model.coefs, dist)[0]

    print(np.average(np.absolute(vals_int_part - vals_int_part_exact)))
    assert np.allclose(vals_int_part_exact, vals_int_part, rtol=0, atol=0.2)
