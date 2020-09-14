#Utils:
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import make_moons, make_circles, make_classification

def generate_data():
    X_train, y_train = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    X_test, y_test = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)


    rng = np.random.RandomState(2)
    
    X_train += 2 * rng.uniform(size=X_train.shape)
    X_test += 3 * rng.uniform(size=X_test.shape)

    linearly_separable_train = (X_train, y_train)
    linearly_separable_test = (X_test, y_test)



    datasets_train = [make_moons(noise=0.1, random_state=0,),
                make_circles(noise=0.1, factor=0.5, random_state=1),
                linearly_separable_train
                ]
    datasets_test = [make_moons(noise=0.3, random_state=10,),
                make_circles(noise=0.3, factor=0.5, random_state=10),
                linearly_separable_test
                ]
    return datasets_train, datasets_test

def one_hot(a):
    h = np.zeros((a.size, a.max()+1))
    h[np.arange(a.size),a] = 1
    return h

def plot_predictions(i,X, Y, X_test, Y_test, pred_train, pred_test, line_x=None, line_y=None, plot_svm=None, plot_nb = None):
    fig = plt.figure(figsize=(15,3))
    
    #TRAIN: indices samples each class
    i_c0 = (Y == 0)
    i_c1 = (Y == 1)
    
    #TRAIN: true and false predictions 
    i_c0_t = (pred_train[i_c0]==0); i_c0_f = (pred_train[i_c0]==1)
    i_c1_t = (pred_train[i_c1]==1); i_c1_f = (pred_train[i_c1]==0)
    
        
    # train
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(X[:,0][i_c0][i_c0_t], X[:,1][i_c0][i_c0_t], marker="+", c='indigo', label='class 0 correct')
    ax1.scatter(X[:,0][i_c0][i_c0_f], X[:,1][i_c0][i_c0_f], marker="o", c='gold', label='class 0 incorrect')
    
    ax1.scatter(X[:,0][i_c1][i_c1_t],X[:,1][i_c1][i_c1_t], marker="+", c='gold', label='class 1 correct')
    ax1.scatter(X[:,0][i_c1][i_c1_f],X[:,1][i_c1][i_c1_f], marker="o", c='indigo', label='class 1 incorrect')
    ax1.set_title(f'Dataset {i}, Train')
    ax1.legend(loc='upper left', ncol=2);
    if plot_svm is not None:
        plot_svc_decision_function(plot_svm, ax1)
    
    if plot_nb is not None:
        plot_nb_decision(plot_nb, ax1)
    
    if not line_x is None and not line_y is None:
        ax1.plot(line_x, line_y)
    
    ax1.set_xlim((min(X[:,0]), max(X[:,0])))
    ax1.set_ylim((min(X[:,1]), max(X[:,1])))
    
    
    
    
    #TEST: indices samples each class
    i_c0 = (Y_test == 0)
    i_c1 = (Y_test == 1)
    
    #TEST: true and false predictions 
    i_c0_t = (pred_test[i_c0]==0); i_c0_f = (pred_test[i_c0]==1)
    i_c1_t = (pred_test[i_c1]==1); i_c1_f = (pred_test[i_c1]==0)
    # test
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(X_test[:,0][i_c0][i_c0_t], X_test[:,1][i_c0][i_c0_t], marker="+", c='indigo', label='class 0 correct')
    ax2.scatter(X_test[:,0][i_c0][i_c0_f], X_test[:,1][i_c0][i_c0_f], marker="o", c='gold', label='class 0 incorrect')
    
    ax2.scatter(X_test[:,0][i_c1][i_c1_t],X_test[:,1][i_c1][i_c1_t], marker="+", c='gold', label='class 1 correct')
    ax2.scatter(X_test[:,0][i_c1][i_c1_f],X_test[:,1][i_c1][i_c1_f], marker="o", c='indigo', label='class 1 incorrect')
    
    if plot_svm is not None:
        plot_svc_decision_function(plot_svm, ax2)
    
    ax2.set_title(f'Dataset {i }, Test ')
    ax2.legend(loc='upper left', ncol=2);
    if not line_x is None and not line_y is None:
        ax2.plot(line_x, line_y)
    
    if plot_nb is not None:
        plot_nb_decision(plot_nb, ax2)
    
    
    ax2.set_xlim((min(X[:,0]), max(X[:,0])))
    ax2.set_ylim((min(X[:,1]), max(X[:,1])))
    
def plot_svc_decision_function(model, ax, plot_support=True):
    """Plot the decision function for a 2D SVC"""

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    
    
    P = model.decision_function(xy).reshape(X.shape)
    
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def plot_nb_decision(model, ax):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                     np.linspace(ylim[0], ylim[1], 81))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    #------------------------------------------------------------
    # Plot the results
    
    #ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary, zorder=2)

    ax.contour(xx, yy, Z, [0.5], colors='k')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax