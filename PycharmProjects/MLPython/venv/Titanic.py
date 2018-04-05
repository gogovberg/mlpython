# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df,test_df]
print(train_df.columns.values)

#Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.

#survival       Survival                0=No, 1=Yes
#pclass         Ticket class            1=1st,2=2nd,3rd
#sex            Sex                     Male,Female
#Age            Age in years
#sibsp          # of siblings/spuses
#parch          # of parents/children
#ticket         Ticket number
#fare           passenger fare
#cabin          Cabin number
#embarked       Port of embarkation     C=Cherbourg, Q=Queenstown, S=Southampton


# 1.Which features are categorical?
#   -Categorical data are variables that contain label values rather than numeric values.
#   -The number of possible values is often limited to a fixed set.
#   -Categorical variables are often called nominal.
#   -Categorical: Survived, Sex, Embarked Ordinal: pclass
# 2.Which features are numerical?
#   -These values change from sample to sample.
#   -Within numerical features are the values discrete, continuous.
#   -Continous: Age,Fare.

print(train_df.head())

# 3.Which features are mixed data types?
#   -Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
#   -Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.

# 4.Which features may contain errors or typos?
#   -Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets,
#    and quotes used for alternative or short names.

print(train_df.tail())

# 5.Which features contain blank, null or empty values?
#   -Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
#   -Cabin > Age are incomplete in case of test dataset.


print('_'*80)
train_df.info()
print('_'*80)
test_df.info()

# 6.What are the data types for various features?
#   -Seven features are integer or floats. Six in case of test dataset.
#   -Five features are strings (object).

print('_'*80)
print(train_df.describe())

# 7.What is the distribution of numerical feature values across the samples?
#   -Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
#   -Survived is a categorical feature with 0 or 1 values.
#   -Around 38% samples survived representative of the actual survival rate at 32%.
#    2224 passengers and crew, 1502 killed ((2224-1502)/2224 ) * 100 = 32%
print('_'*80)
print(train_df.groupby('Survived').count())
print('_'*80)
print(train_df.Survived.value_counts())
#    Survived 342, Died 549, (342/(342+549)) * 100 = 38%
#   -Most passengers (> 75%) did not travel with parents or children.
print('_'*80)
print(train_df.Parch.value_counts())
train_df[train_df['Parch']==0].count()
#    Parch=0 -> 678
#   -Nearly 30% of the passengers had siblings and/or spouse aboard.
print('_'*80)
train_df[train_df['SibSp']>0].count()
#    (283/891) * 100 = 31.7%
#   -Fares varied significantly with few passengers (<1%) paying as high as $512.
print('_'*80)
train_df[train_df['Fare']==train_df.loc[train_df['Fare'].idxmax()]['Fare']].count()
#    0.33% < 1%
#   -Few elderly passengers (<1%) within age range 65-80.
print('_'*80)
train_df[train_df['Age'].between(65,80,inclusive=False)].count()
#   7/891 * 100 = 0.78%



# 8.What is the distribution of categorical features?
print('_'*80)
train_df.describe(include=['O'])
#   -Names are unique across the dataset (count=unique=891)
#   -Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
#   -Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
#   -Embarked takes three possible values. S port used by most passengers (top=S)
#   -Ticket feature has high ratio (22%) of duplicate values (unique=681).

# 9.Assumtions based on data analysis
# a) Correlating
#       -We want to know how well does each feature correlate with Survival. We wantto do this early in our project and
#        match there quick correlations with modelled correlations later in the project.
# b) Completeing
#       1. We may want to complete Age feature as it is definitely correlated to survival.
#       2. We may want to complete Embarked feature as it may correlate with survivl or other feature.
# c) Correcting
#       1. Ticket feature may be dropped from our analysis as it contains high ration of duplicates and there may not be
#          correlation between Ticket and survival
#       2. Cabin feature may be dropped as it highly incomplete or contains many null values
#       3. Passenger ID may be dropped from traning set as it does not contribuite to survival
#       4. Name feature is relatively non standard ,maybe dropped.
# d) Creating
#       1. New feature  called Family  based on Parch and SibSp to get total count
#       2. Extract Title from name feature
#       3. New feature for Age bands.
#       4. New featire Fare rage
# e) Classifying
#       1. Women were more likly to have survived?
#       2. Children were more likely to have survived?
#       3. The upper-class passengers were more likly to have survived




print('_'*80)
print(train_df.describe(include=['O']))
print('_'*80)
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*80)
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*80)
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*80)
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


## Analyze by visualizing data

# Correlating numerical features
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# Observation
#   -Infants (Age <=4) high survivning rate.
#   -Oldest passenger is 80 - survived
#   -Large number of 15-25 years old did not survived
#   -Most passengers are between 15-35 yo
# Decisions
#   -Consider Age in our model traning
#   -Complete the Age feature for null values
#   -We should band age groups

# Correlating numerical and ordinal features
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# Observations
#   -Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
#   -Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
#   -Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
#   -Pclass varies in terms of Age distribution of passengers.
# Decisions
#   -Consider Pclass for model training.

#Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# Observation
#   -Female passengers had much better survival rate than males. Confirms classifying (#1).
#   -Exception in Embarked=C where males had higher survival rate.
#    This could be a correlation between Pclass and Embarked and in turn Pclass and Survived,
#    not necessarily direct correlation between Embarked and Survived.
#   -Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports.
#   -Ports of embarkation have varying survival rates for Pclass=3 and among male passengers
# Decisions
#   -Add Sex feature to model training.
#   -Complete and add Embarked feature to model training.


#Correlating categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# Observation
#   -Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges
#   -Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# Decisions
#   -Consider banding Fare feature.


##Wringle data

#Correcting by dropping features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

#Correcting a new feature extracting from existing

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

crostabSex = pd.crosstab(train_df['Title'], train_df['Sex'])
crostabAge = pd.crosstab(train_df['Title'], train_df['Age'])
crostabSurv = pd.crosstab(train_df['Title'], train_df['Survived'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

tsv_mean = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

#Converting a categorical feature

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#Completing a numerical continuous feature

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0,2):
        for j in  range(0,3):
            # returns list of ages for sex i and pclass j+1 (for every combination)
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            # returns the median age from gues_df
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            # pirely label-location based indexer for selection by label
            # can be used for selecting rows by label/index
            # selecting rows with a boolean/conditional lookup
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
temp_age = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32),'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48),'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64),'Age'] = 3
    dataset.loc[ dataset['Age'] > 64,'Age' ] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# Create new feature combining existing features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

temp_family = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(
    by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

temp_isAlone = train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(
    by='Survived',ascending=False)

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

#create an artificial feature combining Pclass and Age.

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#Completing a categorical feature
# since embarked has two missing values we simply feel them with most common occurence

# drop na values if
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

test_emb = train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(
    by='Survived', ascending=False)

# Converting categorical feature to numeric

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

#Quick completing and converting a numeric feature
# Fill NA/NaN values using the specified method
#
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#We can not create FareBand.
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
temp_fareb=train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(
    by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]


## Model, predict and solve
# we have classification/regression problem  - and we use supervised learning because we give data set to the model
# 1. Logistic Regression
#   - Logistic regression measuers the relationship between the categorical dependent variable (feature)
#     one or more independent variables by estimating probabilities using logistic function, cumulative logistic distributio

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

temp_coeff_df = coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
temp_models = models.sort_values(by='Score', ascending=False)

plt.show()

