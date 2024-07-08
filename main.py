from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.base import clone

from copy import deepcopy

from collections import OrderedDict

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
jax.default_device(jax.devices('cpu')[0])

import jax.numpy as jnp
import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt

from dag_kl import dag_kl
from better_graphs import process_graph

import csv

def visualize(data, y, y_hat):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Double the width to accommodate two plots
    ax[0].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y_hat, edgecolor='k', s=50, cmap='rainbow')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].set_title('PCA, Predicted Labels by GMM')
    ax[0].grid(True)
    scatter = ax[1].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y, edgecolor='k', s=50, cmap='viridis')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].set_title('PCA Ground Truth')
    ax[1].grid(True)
    plt.show()


# I am like 93% sure that this is the part where they check the redundancy condition
def check_epsilons(gmm, n_samples, axes_to_keep, n_clusters):
    # print(n_samples, axes_to_keep)
    data, labels = gmm.sample(n_samples)
    data_castrated = data[:, axes_to_keep]
    # print(np.shape(data), np.shape(data_castrated))

    gmm_castrated = GaussianMixture(n_components=n_clusters, covariance_type='diag')

    gmm_castrated.weights_ = gmm.weights_
    gmm_castrated.covariances_ = gmm.covariances_[:, axes_to_keep]
    gmm_castrated.means_ = gmm.means_[:, axes_to_keep]
    gmm_castrated.precisions_cholesky_ = jnp.sqrt(1 / gmm_castrated.covariances_)
    gmm_castrated.converged = True

    p_L_X = gmm.predict_proba(data)
    p_L_Xs = gmm_castrated.predict_proba(data_castrated)
    # print('p_L_X', p_L_X)
    # print('p_L_Xs', p_L_Xs)

    # find problem
    # go line by line, ask myself what literally each line of code does and why
    # understand what went wrong
    # lots of print statements!

    p_X = np.exp(gmm.score_samples(data))

    # edkl = p_X @ np.einsum("xl,xl->x", p_L_X, (np.log(p_L_X) - np.log(p_L_Xs)))
    # the above line was commented out when it got to me 
    
    # print("X:", p_L_X, type(p_L_X))
    # print("S:", p_L_Xs, type(p_L_Xs))

    p_L_X[p_L_X < 1e-199] = 1e-199
    p_L_Xs[p_L_Xs < 1e-199] = 1e-199
    # need these to avoid catastrophic NaNs later comparing [0, eps, 1-eps*] to [0, 0, 1] because * rounds to 1.

    logdiffs = np.array(np.log(p_L_X) - np.log(p_L_Xs))

    edkl = np.einsum("xl,xl->x", p_L_X, logdiffs).mean()
    # the problem here has to do with the program detecting [0 0 1]-ish things in p_L_X, p_L_Xs.
    # the agreement value between [0 0 1] and [0 0 1] should be 1, not NaN!

    edkl = edkl / jnp.log(2)

    if edkl > 50:
        print('wow this is real bad')

    print("E_x[Dkl(P[(L|X) || (L|Xs)])] = ", edkl)

    return edkl

def main():
    n_gmm_components = 3
    # original was 3 (species of flower); NCEI divides weather stations ~geographically into 6, ultimately I ended up using Kepler Objects instead which... class into Candidates, Confirmeds, and False Positives. F.
    covariance_type = 'diag'
    init_params = 'random' # default: 'kmeans'

    kepler_raw = open(r"cumulative_2024.06.26_22.13.17.csv", "r")
    kepler_str = kepler_raw.read()
    kepler_raw.close()

    kepler_str = kepler_str.split('\n')
    kepler_str = kepler_str[23:-1]  # removing the header stuff and the last blank row
    kepler_pure = []
    for S in kepler_str:
        temp = S.split(',')
        kepler_pure.append(temp)  # I don't know that words can express how annoyed I am with NOAA nor how pleased I am with NASA, for the brevity of this code vs the other one.

    # KepID, Ground Truth, orbital period, transit duration, transit depth, body/star rad ratio, body radius, orbit sMa, inclination, body temperature, body insolation flux, star temperature, star radius, star mass

    kepler_filtered = []
    observations = ''

    for P in kepler_pure:
        temparray = []
        temparray.append(P[0])
        mapnumber = -1
        if P[1] == 'CONFIRMED':
            mapnumber = 0
        elif P[1] == 'CANDIDATE':
            mapnumber = 1
        elif P[1] == 'FALSE POSITIVE':
            mapnumber = 2
        else: print('Oh no! There is a very bad problem! This should literally never print!', mapnumber)

        # P[1] = '\'' + P[1] + '\''
        temparray.append(P[1])
        temparray.append(P[2:])
        kepler_filtered.append(temparray)

        observations = observations+','.join(P[2:])+','+ str(mapnumber) +'\n'
        # print(observations)

    # at this point, kepler_filtered consists of a list of lists of lists
    # A: list of all observation-packages B
    # B: KepID, Ground Truth (confirmed vs candidate vs false positive), list of observational data C
    # C: orbital period, transit duration, transit depth, body/star rad ratio, body radius, orbit sMa, inclination, body temperature, body insolation flux, star temperature, star radius, star mass

    featurenames = 'orb_period,xit_duration,xit_depth,body_r_per_stellar_r_rat,body_r,orb_semimajor,inc,body_temp,body_insolation_flux,stellar_temp,stellar_r,stellar_m,kepler_status'
    fakecsv = featurenames + '\n' + observations

    totallyrealcsv = open('totallyrealcsv.csv', 'w')
    totallyrealcsv.write(fakecsv)
    totallyrealcsv.close()

    # data_df = pd.read_csv("kepler_final.csv")  # WRONG AS IS - just setting data_df = kepler_pure is also wrong.
    data_df = pd.read_csv('totallyrealcsv.csv', dtype=np.float32)  # The NaNs in this, when they exist, come from blanks. In my case there was a single blank orb_sMa entry that was blank, even after I filtered by >0.
    data = data_df[[c for c in data_df.columns[:-1]]].to_numpy(dtype=np.float32)
    y = data_df['kepler_status'].map({"CONFIRMED": 0, "CANDIDATE": 1, "FALSE POSITIVE": 2}).values

    # print(data_df[data_df.isna().any(axis=1)])
    # print(data_df.loc[data_df.isna()])

    gmm = GaussianMixture(n_components=n_gmm_components, random_state=0,
                          covariance_type=covariance_type, init_params=init_params).fit(data)
    y_hat = gmm.predict(data)

    log_probs = jnp.log(gmm.predict_proba(data))
    probs = gmm.predict_proba(data)
    probs[jnp.isinf(log_probs)] = 0.0
    log_probs = log_probs.at[jnp.isinf(log_probs)].set(0.0)
    entropy_l_given_x = -(probs*log_probs).sum(axis=1).mean() / jnp.log(2)
    print("Entropy of P[L|X]: ", entropy_l_given_x)

    for axes_to_keep in dropped_axis_list:
        print(axes_to_keep)
        check_epsilons(gmm, n_samples=len(data), axes_to_keep=axes_to_keep, n_clusters=n_gmm_components)
        # assert False
        

    redundancy_error = 0
    for axes_to_keep in dropped_axis_list:
        redundancy_error += check_epsilons(gmm, n_samples=len(data), axes_to_keep=axes_to_keep, n_clusters=n_gmm_components)
    print("\nSum of redundancy errors for weak invar: ", redundancy_error)

    print("\nIsomorphism bound: ", redundancy_error + entropy_l_given_x*2)

    # visualize(data, y, y_hat)


    print('\n\n=================\n')

    gmm2 = GaussianMixture(n_components=n_gmm_components, random_state=1,
                          covariance_type=covariance_type, init_params=init_params).fit(data)
    y_hat_2 = gmm2.predict(data)

    log_probs_2 = jnp.log(gmm2.predict_proba(data))
    probs_2 = gmm2.predict_proba(data)
    probs_2[jnp.isinf(log_probs_2)] = 0.0
    log_probs_2 = log_probs_2.at[jnp.isinf(log_probs_2)].set(0.0)
    entropy_l_given_x_2 = -(probs_2 * log_probs_2).sum(axis=1).mean() / jnp.log(2)
    print("Entropy of P[L|X]: ", entropy_l_given_x_2)

    for axes_to_keep in dropped_axis_list:
        print(axes_to_keep)
        check_epsilons(gmm2, n_samples=len(data), axes_to_keep=axes_to_keep, n_clusters=n_gmm_components)

    redundancy_error = 0
    for axes_to_keep in dropped_axis_list:
        redundancy_error += check_epsilons(gmm2, n_samples=len(data), axes_to_keep=axes_to_keep, n_clusters=n_gmm_components)
    print("\nSum of redundancy errors for weak invar: ", redundancy_error)

    print("\nIsomorphism bound: ", redundancy_error + entropy_l_given_x_2 * 2)

    print("\n\n==============\n")

    p_L_X_alice = gmm.predict_proba(data)
    p_L_X_bob = gmm2.predict_proba(data)
    p_La_Lb = np.einsum("xa,xb->xab", p_L_X_alice, p_L_X_bob).mean(axis=0)
    p_La_given_lb = p_La_Lb/p_La_Lb.sum(axis=0)
    p_Lb_given_la = p_La_Lb.T / p_La_Lb.T.sum(axis=0)

    entropy_la_given_lb = -(p_La_Lb * jnp.log(p_La_given_lb)).sum() / jnp.log(2)
    entropy_lb_given_la = -(p_La_Lb.T * jnp.log(p_Lb_given_la)).sum() / jnp.log(2)

    print("Entropy L1 | L2: ", entropy_la_given_lb)
    print("Entropy L2 | L1: ", entropy_lb_given_la)

    p_l1 = gmm.predict_proba(data).mean(axis=0)
    p_l2 = gmm2.predict_proba(data).mean(axis=0)

    entropy_l1 = -(p_l1 * jnp.log(p_l1)).sum() / jnp.log(2)
    entropy_l2 = -(p_l2 * jnp.log(p_l2)).sum() / jnp.log(2)
    print("Entropy L1: ", entropy_l1)
    print("Entropy L2: ", entropy_l2)

if __name__ == "__main__":
    print('=====NEW RUN OF LORXUS\'S ADAPTATION OF DAVID LORELL\'S GMM CLUSTERER; NEW PRINTOUT STARTS HERE=====')
    full_axes = list(range(12))  # makes the list of the 12 ways a single axis might be dropped
    # print(full_axes)
    dropped_axis_list = []

    for i in range(12):
        tempfull = full_axes.copy()
        tempfull.remove(i)
        dropped_axis_list.append(tempfull)
        # print(tempfull)

# print(dropped_axis_list)
    main()
