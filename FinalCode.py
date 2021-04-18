#!/usr/bin/env python
# coding: utf-8

# ### Team Name : Data Bots
# ### Team Members: 
# #### Ishaan Thakur(x2020etz, #202004648), Mohit Kapoor(x2020flf, #202005813), Geethasri Rakonda(x2020dxv, #202003910)

# ## **Project - The Toxicity Prediction(Kaggle InClass Competition)**
# #### Datasets Provided: train, test, feamat, etc.
# #### Task : To build a predictive model that gives the toxicity of chemicals.

# ## **Libraries Required**

# In[120]:


import numpy as np   #used for linear algebra and arrays
import pandas as pd  #used for data processing
from numpy import mean
from sklearn.preprocessing import LabelEncoder  #used for preprocess the data - label encoding

from sklearn.model_selection import StratifiedKFold  #used for cross_validation and splitting the data into train data and test 
from sklearn.model_selection import cross_val_score  #used to obtain the cross_val_scores(cross validation evaluation)
from sklearn.metrics import f1_score  #used to obatain the f1_scores - evaluation

from sklearn.feature_selection import RFE   #to obtain best features
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV     #used to perform parameter tuning

from sklearn.pipeline import Pipeline  #to perform multiple tasks at a time while model training

#different classifiers to build prediction model.
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm
from sklearn.svm import SVC

import lightgbm
from lightgbm import LGBMClassifier

#sci-kit learn is a package used for classification, clustering, regression and dimentionality reduction, etc.


# ## **Load Datasets** 

# In[71]:


#load train dataset
train = pd.read_csv('/Users/mohitkapoor/Downloads/InputFiles_Toxic/train.csv')   

#load test dataset
test = pd.read_csv('/Users/mohitkapoor/Downloads/InputFiles_Toxic/test.csv') 

#load feamat dataset 
feature_matrix = pd.read_csv('/Users/mohitkapoor/Downloads/InputFiles_Toxic/feamat.csv')   


# ## **Data Analysis**

# ### **"train" dataset**

# In[72]:


print('train datframe :\n')

print(train.info(), '\n')
print(train.describe(include = 'all'), '\n')

#Are there any "infinite values" in train dataset?
print('Infinite values in "train" dataframe : ', train.isin([np.inf, -np.inf]).values.any())


# ### **"test" dataset**

# In[73]:


print('test datframe :\n')

print(test.info(), '\n')
print(test.describe(include = 'all'), '\n')

#Are there any "infinite values" in test dataset?
print('Infinite values in "test" dataframe : ', test.isin([np.inf, -np.inf]).values.any()) 


# ### **"feamat" dataset**

# In[74]:


print('feature_matrix datframe :\n')

print(feature_matrix.info(), '\n')
print(feature_matrix.describe(include = 'all'), '\n')

#Are there any "null values" and "infinite values" in feamat dataset?
print('Null values in "feature_matrix" dataframe : ',       feature_matrix.isin([np.nan]).values.any())    
print('Infinite values in "feature_matrix" dataframe : ',       feature_matrix.isin([np.inf, -np.inf]).values.any())    


# ## Data Preparation/Data Preprocessing

# ### **Handling infinite values in "feature_matrix"** 

# In[75]:


feature_matrix_new = feature_matrix.copy()  #creating a copy of "feature_matrix" dataframe


# In[76]:


#Function to find "Column_names" & "No of infinite values" in feature_matrix_new dataframe.
def infinite_values(feature_matrix_new):
    count_inf  = feature_matrix_new.isin([np.inf,-np.inf]).sum().sort_values(ascending = False)
    count_inf = count_inf[count_inf > 0]
    return  count_inf
    
#Get the columns which contains infinite values(with their count values) in "feature_matrix_new" dataframe.
print('Column which contains infinite values in "feature_matrix_new" dataframe :\n',       infinite_values(feature_matrix_new))


# In[77]:


print('Rows with infinite values in V15:\n\n',       feature_matrix.loc[:, ['V1', 'V2', 'V15']][feature_matrix['V15'].isin([np.inf,-np.inf])])

#Replace "inf & -inf values" with "nan values" in V15
feature_matrix_new = feature_matrix_new.replace([np.inf, -np.inf], np.nan)

print('\nAfter replacing infinite values in V15 with null values:\n\n',       feature_matrix_new.loc[1996:5716, ['V1', 'V2', 'V15']])


# In[78]:


feature_matrix_new.info()

#Are there any "infinite values" in feature_matrix_new dataframe?
print('\nInfinite values in "feature_matrxi_new" dataframe : ',       feature_matrix_new.isin([np.inf, -np.inf]).values.any())

#Are there any "null values" in feamat dataset?
print('\nNull values in "feature_matrix_new" dataframe : ',       feature_matrix_new.isin([np.nan]).values.any())    

print('\nDescribe column V15: \n', feature_matrix_new['V15'].describe())


# In[79]:


#Filling null values in V15 with its median
feature_matrix_new['V15'].fillna(value = feature_matrix_new['V15'].median(), inplace = True)


# In[80]:


#Are there any "null values" in feature_matrix_new dataframe?
print('\nNull values in feature_matrix_new : ', feature_matrix_new.isna().values.any())   

print('\nDescribe column V15: \n', feature_matrix_new['V15'].describe())


# ### **Split the columns "Id", "x" in "train", "test" dataframes respectively into V1 and aId.**

# In[81]:


train_split = train.copy()  #creating a copy of "train" dataframe    
test_split = test.copy()   #creating a copy of "test" dataframe


# In[82]:


print('train_split datframe :\n')
print(train_split.info(), '\n')

print('test_split dataframe :\n')
print(test_split.info())


# In[83]:


#train_split datframe:

print('Before splitting :\n\n', train.describe(include = 'all'), '\n')

# Adding two new columns to the existing dataframe. 
# splitting is done with respect to ";"
train_split[['V1', 'aId']] = train_split.Id.str.split(';', expand = True) 

print('After splitting into two new columns :\n\n', train_split.describe(include = 'all')) 


# In[84]:


#test_split dataframe 

print('Before splitting :\n\n', test.describe(include = 'all'), '\n')

#Adding two new columns to the existing dataframe. 
#splitting is done with respect to ";"
test_split[['V1' , 'aId']] = test_split.x.str.split(';', expand = True) 

print('After splitting into two new columns :\n\n', test_split.describe(include = 'all')) 


# ### **Merge using left join to form new dataframes "train_feamat" and "test_feamat"**

# In[85]:


#Merge "train_split" and "feature_matrix_new" with respect to column "V1" using left join function
train_feamat = pd.merge(train_split, feature_matrix_new, on = 'V1', how = 'left') 

print(train_feamat)


# In[86]:


# merge "test_split" and "feature_matrix_new" with respect to column "V1" using left join function
test_feamat = pd.merge(test_split , feature_matrix_new , on = 'V1' , how = 'left') 

print(test_feamat)


# In[87]:


print('train_feamat datframe :\n')

print(train_feamat.info(), '\n')
print(train_feamat.describe(include = 'all'), '\n')

#Are there any "null values" in train_feamat dataframe?
print('Null values in train_feamat : ', train_feamat.isna().values.any())  


# In[88]:


print('test_feamat datframe :\n')

print(test_feamat.info(), '\n')
print(test_feamat.describe(include = 'all'), '\n')

#Are there any "null values" in test_feamat dataframe?
print('Null values in test_feamat : ', test_feamat.isna().values.any())  


# ### **Feature Engineering**

# In[89]:


train_fm = train_feamat.copy()  #creating copy of "train_feamat" dataframe
test_fm = test_feamat.copy()    #creating copy of "test_feamat" dataframe


# In[90]:


print('train_fm.shape : ' , train_fm.shape)
print('test_fm.shape : ' , test_fm.shape)


# In[91]:


print('train_fm datframe :\n')
print(train_fm.info(), '\n')
print('dtypes :\n', train_fm.dtypes)

print('\ntest_fm dataframe :\n')
print(test_fm.info(), '\n')
print('dtypes :\n', test_fm.dtypes)


# #### Converting column V1 from object type to int64 type(categorical and assigning numerical values)

# In[92]:


#initializing the LabelEncoder() for column V1
train_V1_LE = LabelEncoder()   
test_V1_LE = LabelEncoder()

#fit and transform the new encoded values into the column 'V1' of dataframes
train_fm['V1'] = train_V1_LE.fit_transform(train_fm['V1'])
test_fm['V1'] = test_V1_LE.fit_transform(test_fm['V1'])


# #### Converting column aId from object type to int64 type

# In[93]:


#converting "aId" from object type to int64 using astype

train_fm['aId'] = train_fm['aId'].astype(int)
test_fm['aId'] = test_fm['aId'].astype(int)


# In[94]:


#each column type of train_fm, test_fm
print('train_fm dtypes :\n\n' , train_fm.dtypes)
print('\ntest_fm dtypes :\n\n' , test_fm.dtypes)

#V1 and aId column values after label encoding 
print('V1 and aId columns of train_fm:\n',train_fm[['Id','V1','aId','V2']])
print('\nV1 and aId columns of test_fm:\n',test_fm[['x','V1','aId','V2']])


# #### dropping "Id" in train_fm and "x" in test_fm

# In[95]:


#columns "Id" in train_fm and "x" in test_fm are object type, unique, redundant and are not required.

train_fm = train_fm.drop(['Id'], axis = 1)
test_fm = test_fm.drop(['x'], axis = 1)


# In[96]:


train_fm.describe(include = 'all')


# In[97]:


test_fm.describe(include = 'all')


# ### Feature Selection

# #### Correlation matrix of train_fm

# In[98]:


#here correlation coefficient takes pearson correlation coefficient value
corr = train_fm.corr()

corr


# #### droping the constant features( having same values i.e either 1 or 0)

# In[99]:


#sub-dataframe of corr with only correlation of "Expected" with all ther columns
corr_exp = corr[0:1] 


# In[100]:


#list of columns that are not correlated and thus not required
fea_nan = (corr_exp.columns[corr_exp.isna().all(0)]).to_list() 

print("Columns in team_fm which are not correlated with 'Expected':\n", fea_nan )


# In[101]:


len(fea_nan)


# In[102]:


train_fm = train_fm.drop( fea_nan , axis = 1 )
test_fm = test_fm.drop( fea_nan , axis = 1 )


# In[103]:


print(train_fm.shape)
print('\n\n',test_fm.shape)


# #### dropping few more columns after observing the correlation coefficient values

# In[104]:


#features with the values 0 and 1 had very less correlation coefficient values
train_fm_01 = train_fm.loc[:,(~train_fm.isin([0,1])).any(axis=0)]
test_fm_01 = test_fm.loc[:,(~test_fm.isin([0,1])).any(axis=0)]


# In[105]:


#these features had very low coefficient value and may not contribute much in the prediction or training
train_fm_new = train_fm_01.drop(['V1','V2','V11', 'V12','V6','V7','V18','V13','V8'], axis = 1)
test_fm_new = test_fm_01.drop(['V1','V2','V11', 'V12','V6','V7','V18','V13','V8'], axis = 1)


# In[106]:


train_fm_new.agg(['count', 'size', 'nunique'])


# In[107]:


test_fm_new.agg(['count', 'size', 'nunique'])


# ### Assigning the features and class label for X and Y parameters

# In[108]:


x = train_fm_new.iloc[:, 1:].values    #assigning x with train dataframe without label column "Expected" 
y = train_fm_new['Expected']     #assigning y with the class label


# In[109]:


print('x:', x.shape, type(x))
print('y:', y.shape, type(y))


# ### Splitting train data(x,y) for training and validation 

# In[110]:


#For splitting train data StaritifiedKFold is used when there is - 
# inbalanced class distribution and the problem is binary classification

folds = StratifiedKFold(n_splits = 15,random_state = None, shuffle = False)

for train_index, test_index in folds.split(x,y):
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]


# In[111]:


print('x_train : ', x_train.shape , '\ny_train : ' , y_train.shape ,       '\nx_test : ' , x_test.shape , '\ny_test : ' , y_test.shape) 


# ## **Model Training - with different classifiers**

# In[112]:


#KNeighbors Classifier
model_knn = KNeighborsClassifier(n_neighbors = 3)
model_knn = model_knn.fit(x_train, y_train)

y_predicted_knn = model_knn.predict(x_test)  #prediction of validation data


# In[44]:


#Naive Bayes 
model_nb = GaussianNB()
model_nb = model_nb.fit(x_train, y_train)

y_predicted_nb = model_nb.predict(x_test)  #prediction of validation data


# In[45]:


#Decision Tree Classifier 
model_dt = DecisionTreeClassifier(random_state = 1)
model_dt = model_dt.fit(x_train, y_train)

y_predicted_dt = model_dt.predict(x_test)  #prediction of validation data


# In[46]:


#Random Forest Classifier
model_rfc = RandomForestClassifier(random_state = 1)
model_rfc = model_rfc.fit(x_train, y_train)

y_predicted_rfc = model_rfc.predict(x_test)  #prediction of validation data


# In[47]:


#SVM Classifier
model_svm = SVC(random_state = 1 )
model_svm = model_svm.fit( x_train , y_train )

y_predicted_svm = model_svm.predict( x_test ) #prediction of validation data


# In[48]:


#Gradient Boosting Classifier
model_gbc = GradientBoostingClassifier(random_state=1)
model_gbc = model_gbc.fit(x_train, y_train)

y_predicted_gbc = model_gbc.predict(x_test)  #prediction of validation data


# In[52]:


#LGBM Classifier
model_lgb = LGBMClassifier(random_state=1)
model_lgb = model_lgb.fit(x_train, y_train)

y_predicted_lgb = model_lgb.predict(x_test)  #prediction of validation data


# ### Performance Evaluation of each model

# #### Performance Evaluation based on f1_macro of validation set

# In[53]:


#F1 score is harmonic mean of precision and recall of a model.

print('knn_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_knn, average = 'macro'))
print('NaiveBayes_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_nb, average = 'macro'))
print('DecisionTree_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_dt, average = 'macro'))
print('RandomForest_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_rfc, average = 'macro'))
print('SVM_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_svm, average = 'macro'))
print('GradientBoosting_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_gbc, average = 'macro'))
print('LGBM_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_lgb, average = 'macro'))


# #### **Performance Evaluation of each model based on cv_score of f1_macro**

# In[123]:


#if cv value is int, stratified cross validation technique is used(appropriate for binary classification problem)

print('knn: ', mean(cross_val_score(model_knn, x, y, cv=10, scoring='f1_macro')))
print('Naive Bayes: ',  mean(cross_val_score(model_nb, x, y, cv=10, scoring='f1_macro')))
print('Decision Tree: ',  mean(cross_val_score(model_dt, x, y, cv=10, scoring='f1_macro')))
print('RandomForest: ', mean(cross_val_score(model_rfc, x, y, cv=10, scoring='f1_macro')))
print('SVM: ', mean(cross_val_score(model_svm, x, y, cv=10, scoring='f1_macro')))
print('GradientBoosting: ', mean(cross_val_score(model_gbc, x, y, cv=10, scoring='f1_macro')))
print('LightGBM: ',mean(cross_val_score(model_lgb, x, y, cv=10, scoring='f1_macro')))


# ## **Final Model Training - with LGB classifier**

# ### Hyperparameter Tuning for LGBM model using GridSearchCV

# In[61]:


#give different values for parameters to be optimized
parameters = {'max_depth': [-1, 10], 'n_estimators' :[2000, 2006], 'max_bin' :[100,255,300], 'metric':['tweedie','binary']}

#model whose parameters are to be tuned
model = LGBMClassifier(random_state=1)

#searching the best parameters
model_lgb_tuned = GridSearchCV(model, parameters, scoring = 'f1_macro', verbose=1)

#fitting to the model
model_lgb_tuned.fit(x_train, y_train)

#scores for best set of optimized parameters 
print(model_lgb_tuned.best_score_)  

#best parameters
print(model_lgb_tuned.best_params_)


# ### Using pipeline with RFE for best features while training the model

# In[113]:


#selecting top best features
fea_selection_lgb = RFE(estimator = LGBMClassifier(max_depth=-1,bagging_fraction =0.5,pre_partition=True ,random_state=31,xgboost_dart_mode=True, silent=True,max_bin=300, metric='tweedie', n_jobs=4, n_estimators=2465), n_features_to_select = 20,)

#lgbm classifier
lgb_model = LGBMClassifier(max_depth=-1, n_estimators=2006, max_bin=300, metric='tweedie', learning_rate =0.1, 
                           feature_fraction=1, pre_partition=False , xgboost_dart_mode=False, n_jobs=4, random_state=31)

#pipeline - feature_selection & model training
model_lgbm = Pipeline(steps = [('best_fea', fea_selection_lgb),('lgb_model', lgb_model)])

#fit the model
model_lgbm.fit(x_train, y_train)


# In[114]:


#transforming the feature selection into x_test
x_test_rfe = fea_selection_lgb.transform(x_test)

#prediction of validation data
y_predicted_lgbm = model_lgbm.predict(x_test) 


# ### **Performance Evaluation**

# #### f1_macro score of validation set

# In[115]:


#F1 score is harmonic mean of precision and recall of a model.
print('LightGradientBoostingClassifier_F1_macro : %1.6f\n' % f1_score(y_test, y_predicted_lgbm, average = 'macro'))


# #### cross_val_score - f1_macro

# In[122]:


#if cv value is int, stratified cross validation technique is used(appropriate for binary classification problem)

cv_f1_scores = cross_val_score(model_lgbm, x, y, cv=10, scoring='f1_macro')
print('LightGradientBoostingClassifier_CV_f1_macro: ', mean(cv_f1_scores))


# ## **Preprocessing test_fm**(actual test dataset used for submission on leaderboard)

# In[66]:


test_fm_new.head()


# In[67]:


#new array of test_fm with best features selected for training
test_sub =test_fm_new.iloc[:, 0:].values 


# ## **Submission**

# In[117]:


#prediction on actual test dataset 
prediction = model_lgbm.predict(test_sub)


# In[118]:


#dataset to be submited
predict_sub = pd.DataFrame({"Id" : test_feamat[ "x" ], "Predicted" : prediction})

predict_sub.to_csv("submission_final_lgb.csv", index=False) 

#generating submission csv file 
print("Submission file generated!!!!")


# In[ ]:




