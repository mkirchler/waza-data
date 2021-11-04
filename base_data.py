import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from pathlib import Path
import os
import json
from functools import partial

from tqdm import tqdm

import numpy as np
from scipy.stats import pearsonr

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, QuantileRegressor, LogisticRegression
from sklearn.svm import LinearSVC


def outlierClassificationFactory(name, d_func, lower, upper):
    f = partial(
            outlierClassification,
            d_func=d_func,
            outlierLower=lower,
            outlierUpper=upper,
            )
    f.__name__ = name
    return f
def outlierRegressionFactory(name, d_func, lower, upper):
    f = partial(
            outlierRegression,
            d_func=d_func,
            outlierLower=lower,
            outlierUpper=upper,
            )
    f.__name__ = name
    return f

def outlierRegression(N, d_func, outlierLower=0.01, outlierUpper=3, seed=123, **kwargs):
    x = d_func(N=N, seed=seed, **kwargs)
    rng = np.random.RandomState(seed+1)
    perc = 0.25 * rng.rand()
    m = int(perc * N)
    ind = rng.choice(N, m)
    length = outlierUpper - outlierLower
    eff_y = outlierLower + length * rng.rand(m)
    eff_x = 1 - length / 6 + rng.rand(m) * length / 3

    x[ind, 0] = x[ind, 0] * eff_x
    x[ind, 1] = x[ind, 1] * eff_y
    return x

def outlierClassification(N, d_func, outlierLower=0.01, outlierUpper=4, seed=123, **kwargs):
    data = d_func(N=N, seed=seed, **kwargs)
    rng = np.random.RandomState(seed+1)
    perc = 0.25 * rng.rand()
    m = int(perc * N)
    ind = rng.choice(N, m)

    length = outlierUpper - outlierLower
    eff = (outlierLower + length * rng.rand(m, 2)) * rng.choice([-1, 1], (m, 2), p=[0.3, 0.7])
    data['x'][ind] *= eff
    data['x1'] = data['x'][data['y']==1]
    data['x2'] = data['x'][data['y']==-1]
    return data
    


def gaussian(N, multiplier, seed=123):
    rng = np.random.RandomState(seed)
    x = rng.randn(N)
    eps = rng.randn(N)
    y = x + multiplier * eps
    return np.array([x, y]).T

def gaussianBlobs(N, n_centers, std=1.0, seed=123):
    x, _ = datasets.make_blobs(
            n_samples=N,
            n_features=2,
            centers=n_centers,
            cluster_std=std,
            random_state=seed,
            )
    return x

# up to 0.15
def rect(N, a=1, b=1, noise=0., seed=123):
    rng = np.random.RandomState(seed)

    x = rng.rand(N, 2) * [a, b]

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    x += noise * rng.randn(N, 2)
    return x

# up to 0.1
def rectEmpty(N, a=1, b=1, noise=0., seed=123):
    rng = np.random.RandomState(seed)

    x = rng.rand(N, 2) #* [a, b]
    empty_side = rng.randint(0, 4, N)
    x[empty_side==0, 0] = 0
    x[empty_side==1, 0] = 1
    x[empty_side==2, 1] = 0
    x[empty_side==3, 1] = 1
    x *= [a, b]

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    x += noise * rng.randn(N, 2)
    return x


def ellipseEmpty(N, A=4, B=1., noise=0., seed=123):
    rng = np.random.RandomState(seed)

    theta = 2 * np.pi * rng.rand(N)
    k = np.sqrt(
            0.9 + 0.2 * rng.rand(N)
            )
    x = np.array([
        A * k * np.cos(theta),
        B * k * np.sin(theta),
        ]).T

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    x += noise * rng.randn(N, 2)
    return x

def ellipse(N, A=4, B=1., noise=0., seed=123):
    rng = np.random.RandomState(seed)

    theta = 2 * np.pi * rng.rand(N)
    k = np.sqrt(rng.rand(N))
    x = np.array([
        A * k * np.cos(theta),
        B * k * np.sin(theta),
        ]).T

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    x += noise * rng.randn(N, 2)
    return x
def moon(N, noise=0., seed=123):
    rng = np.random.RandomState(seed)

    x, y = datasets.make_moons(n_samples=2*N, noise=noise, random_state=seed)
    x = x[y==1]

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    return x


# up to 0.5
def moonsClassification(N, noise=0., seed=123):
    rng = np.random.RandomState(seed)

    x, y = datasets.make_moons(n_samples=N, noise=noise, random_state=seed)
    y = 2 * y - 1

    degrees = rng.randint(360)
    x = rotate_points(x, degrees)
    data = {'x': x, 'y': y, 'x1': x[y==1], 'x2': x[y==-1]}
    return data


def rotate_points(x, degrees):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]])
    return (R @ x.T).T

# noise up to 1
def geoClassification(N, geo='ellipse', lower=0.1, upper=3, seed=123):
    rng = np.random.RandomState(seed)
    if 'ellipse' in geo:
        A = lower + (upper - lower) * rng.rand(2)
        B = lower + (upper - lower) * rng.rand(2)
        noise = 0.1 + 0.4 * rng.rand()
        translation_mult = 3.
        param1 = {'A': A[0], 'B': B[0], 'noise': noise}
        param2 = {'A': A[1], 'B': B[1], 'noise': noise}
    elif 'rect' in geo:
        a = lower + (upper - lower) * rng.rand(2)
        b = lower + (upper - lower) * rng.rand(2)
        noise = 0.15 * rng.rand()
        translation_mult = 1.
        param1 = {'a': a[0], 'b': b[0], 'noise': noise}
        param2 = {'a': a[1], 'b': b[1], 'noise': noise}
    x1 = eval(geo)(N//2, seed=100*seed, **param1)
    x2 = eval(geo)(N//2, seed=200*seed, **param2)
    theta = 2 * np.pi * rng.rand()
    translation = np.array([
        np.cos(theta),
        np.sin(theta),
        ])
    x2 += translation_mult * translation
    x = np.concatenate([x1, x2])
    y = np.ones(N)
    y[N//2:] = -1
    data = {'x': x, 'y': y, 'x1': x1, 'x2': x2}
    return data


def gaussianClassification(N, seed=123):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(N//2, 2)
    x2 = rng.randn(N//2, 2) + 4
    y = np.ones(N)
    y[N//2:] = -1
    x = np.concatenate([x1, x2])
    data = {'x': x, 'y': y, 'x1': x1, 'x2': x2}
    return data


# linear classification


def logistic_regression(data, C=0):
    x = data['x']
    y = data['y']
    clf = LogisticRegression(C=C, penalty='l2' if C>0 else 'none')
    clf.fit(x, y)
    w1, w2 = clf.coef_[0]
    return {'w1': w1, 'w2': w2, 'b': clf.intercept_[0]}

def svc(data, C=1, loss='hinge', penalty='l2'):
    x = data['x']
    y = data['y']
    clf = LinearSVC(C=C, loss=loss, penalty=penalty, max_iter=20000)
    clf.fit(x, y)
    w1, w2 = clf.coef_[0]
    return {'w1': w1, 'w2': w2, 'b': clf.intercept_[0]}

from sklearn.linear_model import RidgeClassifier
def ridge_clf(data, alpha=1):
    x = data['x']
    y = data['y']
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(x, y)
    w1, w2 = clf.coef_[0]
    return {'w1': w1, 'w2': w2, 'b': clf.intercept_[0]}

# correlation 

def correlation(data):
    x = data[:, 0]
    y = data[:, 1]
    cc, _ = pearsonr(x, y)

    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)

    beta = cc * sy / sx
    alpha = my - beta * mx
    return {'cc': cc, 'alpha': alpha, 'beta': beta, 'mx': mx, 'my': my, 'sx': sx, 'sy': sy}

# linear regression

def linear_regression1d(data):
    x = data[:, 0]
    y = data[:, 1]
    reg = LinearRegression()
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_}

def ridge1d(data, alpha):
    x = data[:, 0]
    y = data[:, 1]
    reg = Ridge(alpha=alpha)
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_}
def mean_absolute_regression1d(data):
    x = data[:, 0]
    y = data[:, 1]
    reg = QuantileRegressor(alpha=0)
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_}

from sklearn.decomposition import PCA
def pca1d(data):
    pca = PCA(n_components=1)
    pca.fit(data)
    [[c1, c2]] = pca.components_
    [m1, m2] = pca.mean_
    return {'w': c2 / c1, 'b': m2 - m1 * c2 / c1}

from sklearn.linear_model import HuberRegressor
def huber1d(data, epsilon=1.35, alpha=1):
    x = data[:, 0]
    y = data[:, 1]
    reg = HuberRegressor(alpha=alpha, epsilon=epsilon)
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_}

from sklearn.svm import LinearSVR
def svr1d(data, C=1.0, epsilon=0.1):
    x = data[:, 0]
    y = data[:, 1]
    reg = LinearSVR(C=C, loss='epsilon_insensitive', max_iter=10000, epsilon=epsilon)
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_[0]}
def svrsq1d(data, C=1.0, epsilon=0.1):
    x = data[:, 0]
    y = data[:, 1]
    reg = LinearSVR(C=C, loss='squared_epsilon_insensitive', max_iter=10000, epsilon=epsilon)
    reg.fit(x[:, None], y)
    return {'w': reg.coef_[0], 'b': reg.intercept_[0]}







path = 'data/DATANAME/SEED/N/PARAMS/{algo|data}.json'
# parameter strings need to be ordered !

def dump_data_params(
        d_funcs,
        fn,
        seeds=range(10),
        Ns=[100],
        ):
    all_params = dict()
    for d_func, params, templ in d_funcs:
        keys = list(params[0].keys())
        s = sorted(keys)
        all_params[d_func.__name__] = {
                'seeds': list(seeds),
                'N': Ns,
                'params': params,
                's': s,
                }
        all_params[f'{d_func.__name__}Outlier'] = {
                'seeds': list(seeds),
                'N': Ns,
                'params': params,
                's': s,
                }
    json.dump(all_params, open(fn, 'w'))


        

def build_regression_task():
    regressions = {
            'lr': linear_regression1d,
            'ridge01': partial(ridge1d, alpha=0.1),
            'ridge1': partial(ridge1d, alpha=1.),
            'ridge10': partial(ridge1d, alpha=10.),

            'huber01': partial(huber1d, alpha=0.1, epsilon=1.35),
            'huber1': partial(huber1d, alpha=1., epsilon=1.35),
            'huber10': partial(huber1d, alpha=10., epsilon=1.35),

            'svr01': partial(svr1d, C=0.1, epsilon=0.1),
            'svr1': partial(svr1d, C=1., epsilon=0.1),
            'svr10': partial(svr1d, C=10., epsilon=0.1),

            'svrsq01': partial(svrsq1d, C=0.1, epsilon=0.1),
            'svrsq1': partial(svrsq1d, C=1., epsilon=0.1),
            'svrsq10': partial(svrsq1d, C=10., epsilon=0.1),

            'mae': mean_absolute_regression1d,
            'pca': pca1d,
            'cc': correlation,
                }
    d_funcs = [
            (
                gaussian,
                [
                    {'multiplier': m}
                    for m in np.arange(-1, 1.01, 0.2)
                ],
                'multiplier{multiplier:.4f}',
            ),
            (
                gaussianBlobs,
                [
                    {'n_centers': nc, 'std': std}
                    for nc in [2, 3, 4, 5]
                    for std in [0.5, 1, 2, 4]
                ],
                'n_centers{n_centers:.4f}_std{std:.4f}',
            ),
            (
                ellipse,
                [
                    {'A': A, 'B': 1, 'noise': noise}
                    for A in [0.1, 0.5, 1.5, 2, 3]
                    for noise in [0.01, 0.1, 0.5]
                    ],
                'A{A:.4f}_B{B:.4f}_noise{noise:.4f}',
            ),
            (
                ellipseEmpty,
                [
                    {'A': A, 'B': 1, 'noise': noise}
                    for A in [0.1, 0.5, 1.5, 2, 3]
                    for noise in [0.01, 0.1, 0.25]
                    ],
                'A{A:.4f}_B{B:.4f}_noise{noise:.4f}',
            ),
            (
                rect,
                [
                    {'a': a, 'b': 1, 'noise': noise}
                    for a in [0.1, 0.5, 1, 1.5, 2, 3]
                    for noise in [0.01, 0.1, 0.15]
                    ],
                'a{a:.4f}_b{b:.4f}_noise{noise:.4f}',
            ),
            (
                rectEmpty,
                [
                    {'a': a, 'b': 1, 'noise': noise}
                    for a in [0.1, 0.5, 1, 1.5, 2, 3]
                    for noise in [0.01, 0.1]
                    ],
                'a{a:.4f}_b{b:.4f}_noise{noise:.4f}',
            ),
            (
                moon,
                [
                    {'noise': noise}
                    for noise in [0.1, 0.2, 0.3, 0.4]
                    ],
                'noise{noise:.4f}',
            ),
            ]
    Ns = [100]
    seeds = range(10)
    outlier_lower = 0.01
    outlier_upper = 8
    dump_data_params(
            d_funcs,
            'data/reg_params.json',
            seeds=seeds,
            Ns=Ns,
            )
    for d_func, param_sets, param_templ in d_funcs:
        for params in param_sets:
            param_pth = param_templ.format(**params)
            build_dset(
                    d_func,
                    Ns,
                    regressions,
                    params,
                    seeds=seeds,
                    rounding=5,
                    drop_func=drop_array,
                    param_pth=param_pth,
                    )
            d_func_outlier = outlierRegressionFactory(
                    name=f'{d_func.__name__}Outlier',
                    d_func=d_func,
                    lower=outlier_lower,
                    upper=outlier_upper,
                    )
            build_dset(
                    d_func_outlier,
                    Ns,
                    regressions,
                    params,
                    seeds=seeds,
                    rounding=5,
                    drop_func=drop_array,
                    param_pth=param_pth,
                    )

# def build_correlation_task():
#     d_funcs = [
#             (gaussian,
#                 [
#                     {'multiplier': -.3},
#                     {'multiplier': 1},
#                     {'multiplier': 0.5},
#                 ]),
#                 ]
#     Ns = [100]
#     seeds = range(10)
#     for d_func, param_sets in d_funcs:
#         for params in param_sets:
#             param_pth = f'm{params["multiplier"]:.4f}'
#             build_dset(
#                     d_func,
#                     Ns,
#                     {'cc': correlation},
#                     params=params,
#                     seeds=seeds,
#                     rounding=5,
#                     drop_func=drop_array,
#                     param_pth=param_pth,
#                     )

def build_classification_task():
    classifications = {
            'lr': logistic_regression,

            'svm01': partial(svc, C=0.1),
            'svm1': partial(svc, C=1),
            'svm10': partial(svc, C=10),

            'svmsq01': partial(svc, C=0.1, loss='squared_hinge'),
            'svmsq1': partial(svc, C=1, loss='squared_hinge'),
            'svmsq10': partial(svc, C=10, loss='squared_hinge'),

            'ridgeclf01': partial(ridge_clf, alpha=0.1),
            'ridgeclf1': partial(ridge_clf, alpha=1),
            'ridgeclf10': partial(ridge_clf, alpha=10),
            }
    d_funcs = [
            (
                gaussianClassification,
                [{}],
                '',
                ),
            (
                geoClassification,
                [
                    {'geo': geo, 'lower': 0.1, 'upper': 3}
                    for geo in ['ellipse', 'ellipseEmpty', 'rect', 'rectEmpty']
                ],
                'geo{geo}_lower{lower:.4f}_upper{upper:.4f}',
            ),
            (
                moonsClassification,
                [
                    {'noise': noise}
                    for noise in [0.1, 0.2, 0.35, 0.5]
                ],
                'noise{noise:.4f}',
            ),
            ]
    Ns = [100]
    seeds = range(1000)
    outlier_lower = 0.01
    outlier_upper = 4
    dump_data_params(
            d_funcs,
            'data/clf_params.json',
            seeds=seeds,
            Ns=Ns,
            )
    for d_func, param_sets, param_templ in tqdm(d_funcs):
        for params in param_sets:
            param_pth = param_templ.format(**params)
            # param_pth = ''
            build_dset(
                    d_func,
                    Ns,
                    classifications,
                    params,
                    seeds=seeds,
                    rounding=5,
                    drop_func=drop_clf,
                    param_pth=param_pth,
                    )
            d_func_outlier = outlierClassificationFactory(
                    name=f'{d_func.__name__}Outlier',
                    d_func=d_func,
                    lower=outlier_lower,
                    upper=outlier_upper,
                    )
            build_dset(
                    d_func_outlier,
                    Ns,
                    classifications,
                    params,
                    seeds=seeds,
                    rounding=5,
                    drop_func=drop_clf,
                    param_pth=param_pth,
                    )



def drop_clf(data, fn, rounding=4):
    for xn in ['x1', 'x2']:
        x = data[xn]
        x = x if rounding is None else np.array(x).round(rounding)
        x = list(list(y) for y in x)
        json.dump(x, open(str(fn).format(x=xn[-1]), 'w'))

def drop_array(data, fn, rounding=4):
    data = np.array(data)
    ind = data[:, 0].argsort()
    data = data[ind]

    data = data if rounding is None else np.array(data).round(rounding)

    data = list(list(x) for x in data)
    json.dump(data, open(str(fn).format(x=''), 'w'))


def standardize_data(data):
    if isinstance(data, dict):
        x = StandardScaler().fit_transform(data['x'])
        data['x'] = x
        data['x1'] = x[data['y']==1]
        data['x2'] = x[data['y']==-1]
    else:
        data = StandardScaler().fit_transform(data)
    return data



from sklearn.preprocessing import StandardScaler
def build_dset(
        d_func,
        Ns,
        models,
        params={},
        seeds=None,
        rounding=5,
        base_dir=Path('data'),
        drop_func=drop_array,
        param_pth='',
        ):
    if seeds is None:
        seeds = range(100)
    for N in Ns:
        for seed in tqdm(seeds):
            data = d_func(N, **params, seed=seed)
            data = standardize_data(data)
            direc = base_dir / d_func.__name__ / str(seed) / str(N) / param_pth
            os.makedirs(direc, exist_ok=True)
            fn = direc / 'data{x}.json'
            drop_func(data, fn, rounding=rounding)
            for model_name, model in models.items():
                fitted = model(data)
                # print(fitted)
                fn = direc / (model_name + '.json')
                json.dump(fitted, open(fn, 'w'))
