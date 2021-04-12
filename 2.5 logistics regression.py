# -*- coding: utf-8 -*-
"""
@author: ellen liu
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import statsmodels.api as smodel
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import chi2


contPred = ['Speed_sensor',	'Engine_Load', 'Intake_Pressure',	'Engine_RPM', 'Intake_Air', 'Flow_Rate', 'Throttle_Pos',	'Ambient', 'Accel','GPS_Bearing','GPS_Altitude',	'Trip_Distance','Litres_Per_km']
catTarget = 'Maintenance_flag'

inputData = pandas.read_csv('C:\\Users\\elena\\Documents\\spyder\\C_DataMining\\truck.csv')

X = inputData[contPred]
y = inputData[catTarget]

y.value_counts(normalize = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, stratify = y)


# Train a multinomial logistic regression model
# The SWEEP Operator
def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, positive integer
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    A = numpy.array(inputM, copy = True, dtype = numpy.float)
    diagA = numpy.diagonal(A)
 
    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = numpy.zeros(pDim)
            ANext[k, :] = numpy.zeros(pDim)
        A = ANext

    return (A, aliasParam, nonAliasParam)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):

   # Find the non-redundant columns in the design matrix fullX
   nFullParam = fullX.shape[1]
   XtX = numpy.transpose(fullX).dot(fullX)
   invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-13)

   # Build a multinomial logistic model
   X = fullX.iloc[:, list(nonAliasParam)]
   logit = smodel.MNLogit(y, X)
   thisFit = logit.fit(method='newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
   thisParameter = thisFit.params
   thisLLK = thisFit.llf
   thisDF = (thisFit.J - 1) * thisFit.K

   # Return model statistics
   return (thisLLK, thisDF, thisParameter, thisFit)


# Begin Forward Selection
X0_train = None

nPredictor = len(contPred)
stepSummary = pandas.DataFrame()

# Intercept only model
X0_train = smodel.add_constant(X_train, prepend = True)[['const']]
llk0, df0, beta0, thisFit = build_mnlogit(X0_train, y_train)
stepSummary = stepSummary.append([['Intercept', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN]], ignore_index = True)

contName = contPred.copy()
entryThreshold = 0.05

for step in range(nPredictor):
   enterName = ''
   stepName = numpy.empty((0), dtype = str)
   stepStats = numpy.empty((0,5), dtype = float)

   # Enter the next predictor
   for X_name in contName:
      X1_train = inputData[[X_name]]
      X1_train = X0_train.join(X1_train)
      llk1, df1, beta1, thisFit = build_mnlogit(X1_train, y_train)
      devChiSq = 2.0 * (llk1 - llk0)
      devDF = df1 - df0
      devPValue = chi2.sf(devChiSq, devDF)
      stepName = numpy.append(stepName, numpy.array([X_name]), axis = 0)
      stepStats = numpy.append(stepStats,
                               numpy.array([[df1, llk1, devChiSq, devDF, devPValue]]), axis = 0)

   # Find a predictor to enter, if any
   minPValue = 1.1
   minI = -1
   for i in range(stepStats.shape[0]):
      thisPValue = stepStats[i,4] 
      if (thisPValue < minPValue):
         minPValue = thisPValue
         minI = i

   if (minPValue <= entryThreshold):
      enterName = stepName[minI]
      addList = [enterName]
      for v in stepStats[minI,:]:
         addList.append(v)
      stepSummary = stepSummary.append([addList], ignore_index = True)
      df0 = stepStats[minI,0]
      llk0 = stepStats[minI,1]

      iInt = -1
      try:
         iInt = contName.index(enterName)
         X1_train = inputData[[enterName]]
         X0_train = X0_train.join(X1_train)
         contName.remove(enterName)
      except ValueError:
         iInt = -1
   else:
      break

   # Print debugging output
   print('Step = ', step+1)
   print('Step Statistics:')
   print(stepName)
   print(stepStats)
   print('Enter predictor = ', enterName)
   print('Minimum P-Value =', minPValue)
   print('\n')


stepSummary = stepSummary.rename(columns = {0:'Predictor', 1:'ModelDF', 2:'LLK', 
                                            3:'DevStat', 4:'DevDF', 5:'DevSig'})
# End of forward selection

u = y_train.astype('category')
y_cat = u.cat.categories

nPredictor = len(contPred)


llk1, df1, beta1, modelMNL = build_mnlogit(X_train, y_train)
#beta1 = beta1.rename(columns = {0: y_cat[1]})


print(modelMNL.summary())

ppY_train_MNL = modelMNL.predict(X_train)
ppY_test_MNL = modelMNL.predict(X_test)


def BinaryMetric (y, y_cat, predProbY, ppY_threshold):
   predClass = numpy.where(predProbY >= ppY_threshold, y_cat[1], y_cat[0])
   MCE = numpy.mean(numpy.where(predClass == y, 0, 1))
   error = numpy.where(y == y_cat[1], 1.0 - predProbY, -predProbY)
   RASE = numpy.sqrt(numpy.mean(error * error))
   AUC = metrics.roc_auc_score(y, predProbY)
   return (MCE, RASE, AUC) 
   
ppY_threshold = numpy.mean(y_train)
print('Observed Proportion of ' + catTarget + ' = 1:', ppY_threshold)

# Compare the metrics of both models on training partition
MCE_train_MNL, RASE_train_MNL, AUC_train_MNL = BinaryMetric(y_train, y_cat, ppY_train_MNL[1], ppY_threshold)
MCE_test_MNL, RASE_test_MNL, AUC_test_MNL = BinaryMetric(y_test, y_cat, ppY_test_MNL[1], ppY_threshold)

# Accuracy
# Determine the predicted class of Y
predY_train = numpy.empty_like(y_train)
nY_train = y_train.shape[0]
for i in range(nY_train):
    if (ppY_train_MNL.iloc[i,1] > 0.5):
        predY_train[i] = 1
    else:
        predY_train[i] = 0
        
predY_test = numpy.empty_like(y_test)
nY_test = y_test.shape[0]
for i in range(nY_test):
    if (ppY_test_MNL.iloc[i,1] > 0.5):
        predY_test[i] = 1
    else:
        predY_test[i] = 0

Accuracy_train = accuracy_score(y_train, predY_train)
Accuracy_test = accuracy_score(y_test, predY_test)

MR_train = 1-Accuracy_train
MR_test = 1-Accuracy_test


# Compare ROC curves of both models on training partition
fpr_train_MNL, tpr_train_MNL, thresh_train_MNL = metrics.roc_curve(y_train, ppY_train_MNL[1], pos_label = y_cat[1])


plt.plot(fpr_train_MNL, tpr_train_MNL, marker = '+', color = 'royalblue', label = 'Logistic')
plt.plot([0,1], [0,1], linestyle = 'dashed', color = 'gray')
plt.title('ROC Curve on Training Partition')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.yticks(numpy.arange(0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend(loc = 'lower right')
plt.show()




dt = tree.DecisionTreeClassifier(
        criterion = 'entropy', max_depth = 5,
        min_samples_leaf = 5, random_state = 60637)
dt.fit(X_train, y_train)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances_dt = pandas.Series(
        dt.feature_importances_,
        index = X.columns)

# Sort importances_rf
sorted_importances_dt = importances_dt.sort_values(ascending=False)

sns.barplot(x=sorted_importances_dt,y=sorted_importances_dt.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Importance Featurea')
