import numpy as np
import pandas as pd
import random
import os
from os import path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb

os.chdir('G:\MBA\PPDA')

#read train data
train=pd.read_csv('TrainingData.csv')
train.columns

#Data preparation for incorrect values by changing them to missing
train2=train.replace({'missing': None, 'na':None, 'NaN':None})
train2.isna().sum()

n=['application_key', 'mvar1', 'mvar2', 'mvar3', 'mvar4', 'mvar5', 'mvar6',
       'mvar7', 'mvar8', 'mvar9', 'mvar10', 'mvar11', 'mvar12', 'mvar13',
       'mvar14', 'mvar15', 'mvar16', 'mvar17', 'mvar18', 'mvar19', 'mvar20',
       'mvar21', 'mvar22', 'mvar23', 'mvar24', 'mvar25', 'mvar26', 'mvar27',
       'mvar28', 'mvar29', 'mvar30', 'mvar31', 'mvar32', 'mvar33', 'mvar34',
       'mvar35', 'mvar36', 'mvar37', 'mvar38', 'mvar39', 'mvar40', 'mvar41',
       'mvar42', 'mvar43', 'mvar44', 'mvar45', 'mvar46']

#mean imputation for missing values using SimpleImpute
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(np.array(train2[n]))
train3=imp_mean.transform(train2[n])

train3=pd.DataFrame(train3)
train3[['mvar47', 'mvar48','default_ind']]=train[['mvar47', 'mvar48','default_ind']]

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
train3['mvar47']= le.fit_transform(train3['mvar47']) 
train3.columns=train.columns

#final train data to build model
df=pd.DataFrame(train3)

#read Test data
test=pd.read_csv('testX.csv')
test.columns

#missing values in test data
test2=test.replace({'missing': None, 'na':None, 'NaN':None})
test2.isna().sum()

#mean impuation  for test data
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(np.array(test2[n]))
test3=imp_mean.transform(test2[n])

test3=pd.DataFrame(test3)
test3[['mvar47', 'mvar48']]=test[['mvar47', 'mvar48']]

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
test3['mvar47']= le.fit_transform(test3['mvar47']) 
test3.columns=test.columns

#final data for making final test predictions
final_data  = pd.DataFrame(test3)

#Dividing train into response and predictors
X = df.iloc[:, 1:-2]
y = df.iloc[:, -1]

#splitting train data for building model and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

#lightgbm dataset creation
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#providing paremeter for lightGBM
params = {
          'task': 'train',
        'objective': 'binary',
        'metric': 'binary_error',
#         'boosting_type': 'dart',    
        'learning_rate': 0.25,      
        'max_depth': 8,             
        'min_data_in_leaf': 5,      
        'max_bin' : 100,            
        'num_leaves' : 12,         
        'num_iteration': 200,       
        'verbose': 1 }

lgbmodel = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_eval], early_stopping_rounds=50)


# predict train output
y_pred = lgbmodel.predict(X_test)
y_pred = {'pred_score': y_pred}
y_pred = pd.DataFrame(y_pred)
y_pred['pred'] = y_pred['pred_score'].apply(lambda x: 1 if x > 0.3 else 0)
y_pred = y_pred['pred']


#check scores
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity=tp/(tp+fn)
avg=(specificity+sensitivity)/2
print("balanced_acuracy:", avg)
print("specificity:", specificity)
print("sensitivity:", sensitivity)
from sklearn.metrics import accuracy_score
print("accuracy:", accuracy_score(y_pred,y_test))
print("f1 score:", f1_score(y_test, y_pred, average='binary'))
print(classification_report(y_test, y_pred))
print("True negative:", tn,"False Positive:", fp)
print("False negative:", fn,"True Positive", tp)

#Get Output
X_final = final_data.iloc[:, 1:-1]
y_pred_final = lgbmodel.predict(X_final)
y_pred_final = {'pred_score': y_pred_final}
y_pred_final = pd.DataFrame(y_pred_final)
y_pred_final['pred'] = y_pred_final['pred_score'].apply(lambda x: 1 if x > 0.3 else 0)
y_pred_final=y_pred_final.drop('pred_score', axis=1)
y_pred_final.to_csv("final_predictions_lgbm_005.csv", index=False)
