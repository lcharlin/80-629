"""Plotting helpers for the MATH 60629 week-4 tutorial.

Students: you do NOT need to read or understand this file. It contains
the plotting code used by the tutorial, so that the notebook can focus
on machine learning rather than on matplotlib. Curious readers are of
course welcome to peek.
"""

import matplotlib.pyplot as plt


def plot_avg_rentals(series, xlabel=''):
    """Bar plot of a pandas Series (e.g., the output of a groupby().mean())."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(series)), series.values, color='#4C72B0')
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index, rotation=0 if len(series) <= 25 else 90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('average rentals')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_train_test(param_values, train_errors, test_errors,
                    xlabel='', ylabel='error', logx=False):
    """Train and test error (or accuracy) as a function of a hyperparameter."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(param_values, train_errors, 'o-', label='train', color='#4C72B0')
    ax.plot(param_values, test_errors, 'o-', label='test', color='#C44E52')
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, max_points=500):
    """Scatter plot of predicted vs. actual values, with the ideal diagonal."""
    n = min(len(y_true), max_points)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(list(y_true)[:n], list(y_pred)[:n], s=12, alpha=0.5,
               color='#4C72B0')
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, '--', color='#C44E52', label='perfect predictions')
    ax.set_xlabel('actual rentals')
    ax.set_ylabel('predicted rentals')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()
