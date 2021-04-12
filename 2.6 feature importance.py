
"""
@author: ellen
"""
# Basic imports
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\elena\\Documents\\spyder\\C_DataMining\\truck.csv', delimiter = ",")


df = pd.get_dummies(df)
X, y = df.drop(['Maintenance_flag'],axis=1),df['Maintenance_flag']


# Set seed for reproducibility
SEED = 2
# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = \
train_test_split(X, y,
test_size=0.3,
random_state=SEED)


dt = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_leaf = 5, random_state = 60637)
modelTree = dt.fit(X_train, y_train)


# Create a pd.Series of features importances
importances_dt = pd.Series(dt.feature_importances_,
index = X.columns)

# Sort importances_rf
sorted_importances_dt = importances_dt.sort_values(ascending=False)

sns.barplot(x=sorted_importances_dt,y=sorted_importances_dt.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Importance Featurea')