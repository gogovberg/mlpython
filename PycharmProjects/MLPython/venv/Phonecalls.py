# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('calls.csv')


train_df.head()
print(train_df.describe())
train_df.Callers = train_df.Callers.astype(str)
train_df.Called = train_df.Called.astype(str)
print(train_df.describe())


train_df['Direction'] = train_df.Callers == "38670216064"


train_df["CalledNetwork"] = train_df["CalledNetwork"].str.replace("Incoming-", "")
train_df["CalledNetwork"] = train_df["CalledNetwork"].str.replace("Outgoing-", "")
print(train_df.CalledNetwork.unique())

fig, ax = plt.subplots()
train_df.CalledNetwork.value_counts().plot(ax=ax, kind='bar')

grid = sns.FacetGrid(train_df, col='Direction', row='CalledNetwork', size=2.2, aspect=1.6)
grid.map(plt.hist, 'CallDuration', alpha=.5, bins=20)
grid.add_legend()


outgoing_calls = train_df.loc[train_df['Direction'] == True]
grouped_network = outgoing_calls.groupby(["CalledNetwork"]).size()
grouped_sum_network = outgoing_calls[["CalledNetwork", 'CallDuration']].groupby(["CalledNetwork"]).sum()
plt.show()

