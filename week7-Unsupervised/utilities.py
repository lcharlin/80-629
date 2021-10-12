###
# Classic libraries
###

import pandas as pd
import numpy as np

###
# Data science librarie
###

import sklearn as sk
from sklearn.cluster import KMeans   # KMeans function
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs  # Easy simulations
from sklearn.model_selection import train_test_split   # Cross validation library

###
# data visualization libaries
###

import matplotlib.pyplot as plt
import seaborn as sns   # A must! For nice an easy figures - look for sns command in the notebook
from matplotlib.pyplot import cm   # This is the color chart that I personnaly prefer


def color(label, dim):
    x = np.linspace(0.0, 1.0, 100)
    a = plt.get_cmap("tab10")(x)[np.newaxis, :, :3][0][0]
    b = plt.get_cmap("tab10")(x)[np.newaxis, :, :3][0][10]
    
    if dim == 4:
        c = plt.get_cmap("tab10")(x)[np.newaxis, :, :3][0][20]
        d = plt.get_cmap("tab10")(x)[np.newaxis, :, :3][0][30]
    
        chart = [a, b, c, d]
    else:
        chart = [a, b]
    
    return(np.dot(label, chart))


def super_scat_it(X, y, dim, clusters_center=0, task='kmeans', wcolor=True):
    
    sns.set(rc={'figure.figsize':(8,6)})
    sns.set(font_scale = 1.5)
    cmap = plt.get_cmap("tab10")
    plt.gca().set_aspect(1)


    #color=cmap.rainbow(np.linspace(0,1, dim))

    if task == 'kmeans':
        
        data = np.concatenate((X, y.reshape(len(y),1)), axis=1)
        ens = pd.DataFrame(data)
        ens.columns = ['x1', 'x2', 'y']
        
        for k in range(dim):
            if wcolor:
                plt.scatter(ens[ens['y']==k]['x1'],ens[ens['y']==k]['x2'], color=cmap(k), label=f'Distribution {k+1}')
            else:
                plt.scatter(ens[ens['y']==k]['x1'],ens[ens['y']==k]['x2'], color='b', label=f'Distribution {k+1}')

        ###
        # Plot presentation
        ###
        if max(abs(y)) > 1:
            plt.scatter(ens[ens['y']== 100]['x1'],ens[ens['y']== 1000]['x2'],color="w", marker="x", label=' ')

        if np.sum(abs(clusters_center)) != 0:
            plt.scatter(clusters_center[:,0], clusters_center[:,1], marker="*", color="k", s=200, label='Clusters center')

    if task == 'EM':
        plt.scatter(X[:,0], X[:,1], color=color(y, dim), s=10)


    # Axes
    plt.xlabel('x$_1$')
    plt.ylabel('x$_2$')
    

    # Ghosting the legend
    leg = plt.gca().legend(loc='center left', bbox_to_anchor=(1, .85))
    leg.get_frame().set_alpha(0)


def distance(data, cluster_centers):
    """
    Description:      
        for each observation, calculate the euclidiean distance of the neerest cluster centers
        return the average distance
    Args:
        X: unlabbeled data
        cluster_centers: cluster centers 
    Return:
        average distance
    """
       
    matrice = np.zeros((data.shape[0], cluster_centers.shape[0]))

    for k in np.arange(cluster_centers.shape[0]):
        matrice[:,k] = np.sum((data - cluster_centers[k,:])**2, axis=1)
        
    return(matrice)


def initiate(data, k, seed=None):
    """
    Description: Function for randomnly initiate cluster centers
    Args:
        data: unlabbeled data
        k: number of cluster 
    Return:
        Initial values of the cluster centers
    """
        
    X1_min=np.min(data[:,0])
    X1_max=np.max(data[:,0])
    
    X2_min=np.min(data[:,1])
    X2_max=np.max(data[:,1])

    if seed != None:
        np.random.seed(seed)
    X1_means = np.random.uniform(X1_min, X1_max, k)
    X2_means = np.random.uniform(X2_min, X2_max, k)
    
    return (np.concatenate((X1_means.reshape((k,1)) , X2_means.reshape((k,1))), axis=1)) 


def estimate_centroid(data, labels):
    """
    Description: Estimate the centroid according to the label associated with each observation
    Args:
        data: unlabeled data
        labels: labal associated to each observations
    Return:
        k Centroids
    """
    
    data = np.concatenate((data, labels.reshape(len(labels),1)), axis=1)
    data = pd.DataFrame(data)
    data.columns = ['x1', 'x2', 'y']
    
    data_0 = np.mean(data[data['y']==0].values, axis=0)
    data_1 = np.mean(data[data['y']==1].values, axis=0)
    
    return(np.array([data_0, data_1])[:,0:2])
