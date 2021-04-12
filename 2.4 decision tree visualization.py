# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:06:39 2021

@author: elena
"""


"""
@author: ellen
"""
# Basic imports
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree
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



import graphviz
dot_data = tree.export_graphviz(modelTree,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['Speed_sensor',	'Engine_Load', 'Intake_Pressure',	'Engine_RPM', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos',	'Ambient', 'Accel','GPS_Bearing','GPS_Altitude',	'Trip_Distance','Litres_Per_km'],
                                class_names = ['No Maintainance', 'Maintainance'])

graph = graphviz.Source(dot_data)
graph

graph.render('C:\\Users\\elena\\Documents\\spyder\\C_DataMining\\tree.png', format = 'png')