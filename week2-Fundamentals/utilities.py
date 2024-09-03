import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def data_simulation(sample_size, scale, period, variance):
    
    x = np.random.uniform(-scale, scale, sample_size)
    x.sort()
    noise = np.random.normal(0, variance, sample_size)
    y = x**1 * np.cos(x / period) + noise
    
    return x, y

def scatter_plot(x_train, x_test, y_train, y_test):
    plt.figure(figsize=(15,5))
    plt.plot(x_train, y_train, '.', color='black', markersize= 3, label='train')
    plt.plot(x_test, y_test, '.', color='red', markersize= 8, label='test')

    plt.xlabel('X')
    plt.ylabel('Y')
    leg = plt.legend(loc='lower center', fontsize='large')
    leg.get_frame().set_alpha(0)
    return plt
    
def plot_polynomial_curves(x_train, x_test, y_train, y_test, degree, scale):

    loss_train_stack, loss_test_stack = [], []
    color=cm.rainbow(np.linspace(0,1,len(degree)))
    plt = scatter_plot(x_train, x_test, y_train, y_test)

    for k,c in zip(range(len(degree)),color):
        coef = np.polyfit(x_train, y_train, degree[k])
        
        y_hat_train = np.polyval(coef, x_train)
        y_hat_test = np.polyval(coef, x_test)
       
        loss_train_stack.append(MSE(y_hat_train, y_train))
        loss_test_stack.append(MSE(y_hat_test, y_test))
        
        print('Polynomial degree: ', degree[k], ' | MSE train:', np.round(loss_train_stack[-1], 4), ' | MSE test:', np.round(loss_test_stack[-1], 4))
        x_draw = np.linspace(-scale, scale, num=200)
        y_draw = np.polyval(coef, x_draw)
        plt.plot(x_draw, y_draw, color=c, label=degree[k],)
        plt.ylim(min(min(y_train), min(y_test)), max(max(y_train), max(y_test)))
        #plt.plot(x_train, y_hat_train, color=c, label=degree[k],)
    
    leg = plt.gca().legend(loc='center left', bbox_to_anchor=(1, .65), title="Polynomial degree of  \n  the fitted curve \n")
    leg.get_frame().set_alpha(0)    

    
def plot_optimal_curve(optimal_train, optimal_test, H_train, H_test, optimal_degree):

    cmap = plt.get_cmap("tab10")   # Because I prefer this color map

    H = np.concatenate((optimal_train, optimal_test, H_train[:,2], H_test[:,2]), axis=0)
    mini, maxi = min(H), max(H)
    linewidth = 2
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)#, layout='constrained')#, figsize=(30,5))
    #ax2 = ax1.twinx()
    #ax2 = ax[1]
    
    # Home made tricks for great legend
    ax1.plot([-20] * len(optimal_train),  label='Test', color='k', linewidth=1)
    ax1.plot([-20] * len(optimal_train), label='Training', linestyle='dashed', color='k', linewidth=1)
    ax1.plot([-20] * len(optimal_train), label='     ', linestyle='dashed', color='white')

    ax1.plot(H_train[:,2], color=cmap(1), linestyle='dashed', linewidth=linewidth)   # Since we are interested in the cubic curve
    ax1.plot(H_test[:,2], color=cmap(1), label='Cubic',linewidth=linewidth)
    ax1.set_xlabel('Sample size:  $\ \log_{10}(n) - 1$')
    ax1.set_ylabel('MSE')
    ax1.set_ylim(mini-50, maxi+100)
    ax1.plot(optimal_train, color=cmap(0), linestyle='dashed', linewidth=linewidth)   # Since we are 'also' interested in the optimal curve
    ax1.plot(optimal_test, color=cmap(0), label='Optimal Capacity', linewidth=linewidth)


    plt.xticks([0, 1, 2, 3, 4, 5])
    leg1 = ax1.legend(loc='center left', bbox_to_anchor=(1.1, .8))   # Legend location is somehow important to me
    leg1.get_frame().set_alpha(0)   # Legend without frame > legend with frame imo

    #plt.show()

    ax2.plot(optimal_degree, color=cmap(2), label='Optimial degree', linewidth=linewidth)   # Optimal degree with respect to the sample size
    ax2.set_ylabel('Degree of the polynomial', fontsize=12, color = 'green')
    ax2.set_xlabel('Sample size:  $\ \log_{10}(n) - 1$')
    leg2 = ax2.legend(loc='center left', bbox_to_anchor=(1.1, .8))   # Legend location is somehow important to me
    leg2.get_frame().set_alpha(0)   # Legend without frame > legend with frame imo

    plt.show()




def train_poly_and_see(sample_size, scale, period, variance, degree):
    H_train = np.zeros((len(sample_size), len(degree)))
    H_test = np.zeros((len(sample_size), len(degree)))

    optimal_train, optimal_test, optimal_degree = [], [], []

    i = 0
    for n in sample_size:

        x_train, y_train = data_simulation(n, scale, period, variance)
        x_test, y_test = data_simulation(1000, scale, period, variance)

        j = 0
        for k in degree:
            coef = np.polyfit(x_train, y_train, k)

            y_hat_train = np.polyval(coef, x_train)
            y_hat_test = np.polyval(coef, x_test)

            H_train[i, j] = MSE(y_train, y_hat_train)
            H_test[i, j] = MSE(y_test, y_hat_test)
            j += 1

        optimal_degree.append(degree[np.argmin(H_test[i, :])])
        optimal_train.append(H_train[i, np.argmin(H_test[i, :])])
        optimal_test.append(H_test[i, np.argmin(H_test[i, :])]) 
        i +=1
    
    return H_train, H_test, optimal_train, optimal_test, optimal_degree
    
def MSE(a, b):
    return ((a-b)**2).mean()
