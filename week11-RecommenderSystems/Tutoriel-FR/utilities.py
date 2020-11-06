import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def top(vector, maximum, k):
    c = maximum * np.argsort(scores)[-k:] + (1 - maximum) * np.argsort(scores)[:k]
    d = []
    for i in np.arange(len(c)):
        d.append(vector[c[i]])
    return d

def rearrange(items, ratings):
    attribute, scores = [], []
    ranking = np.argsort(ratings)

    for k in np.arange(len(ranking)):
        attribute.append(items[ranking[k]])
        scores.append(ratings[ranking[k]])

    return attribute, scores

def convert(data, nb_users, nb_movies):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

def inner_concatenation(a, b):
    c = []
    for k in np.arange(len(a)):
        c.append(a[k]+b[k])
    return c

def transform(user_sex, user_age, user_occupation):

    argument = users_occupation_name.tolist().index(user_occupation)
    empty = np.zeros(len(users_occupation_name))
    empty[argument] = 1

    if user_sex == 'Male':
        user_sex = 0
    else:
        user_sex = 1

    return np.concatenate(([user_sex], [user_age], empty.T)).T.tolist()

def descriptive(variable):

    if variable.describe().dtype == 'float64':
        stats = pd.DataFrame(np.matrix([np.round(np.mean(variable), 2), np.round(np.std(variable), 2), np.min(variable), np.percentile(variable, 25),
        np.round(np.median(np.array(variable)), 2), np.percentile(variable, 75), np.max(variable)]))
        stats.columns = np.array(['Mean', 'Std', 'Min', '1 qrt', 'Med', '3 qrt', 'Max'])
    else:

        stats = pd.DataFrame(np.matrix((variable.describe().freq / len(variable) * 100)))
        stats.columns = np.array(['Proportion'])
    return stats

def barplot(attribute, scores, xlabel, ylabel, title, rotation):

    label = np.array(attribute)[np.array(scores) != 0]
    scores = np.array(scores)[np.array(scores) != 0]

    sns.set(rc={'figure.figsize':(6,4)})
    sns.set(font_scale = 2)

    plt.bar(label, scores)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(label, label, fontsize=10, rotation=rotation)
    plt.title(title)

    plt.show()

def freq_analysis(names, frequency, recommendations):

    freq = []
    for k in np.arange(len(recommendations)):
        movie_id = names.tolist().index(recommendations[k])
        freq.append(frequency[movie_id])

    return np.array(freq)
