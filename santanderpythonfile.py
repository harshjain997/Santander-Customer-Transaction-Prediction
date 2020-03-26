#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction
# ---

# In this challenge, we need to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.

# ### File descriptions
# 
# * **train.csv** - the training set.
# * **test.csv** - the test set. The test set contains some rows which are not included in scoring.
# Load all required packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numba import jit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
import os

%matplotlib inline



# ## Data Exploration
# ---

# In[2]:


#Read train& test data
os.chdir(r"C:\Users\harsh\Desktop\harsh jain\santander")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[54]:



train.head(10)


# In[55]:


test.head(10)


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[16]:


train.shape, test.shape


# ### Correlation analysis
# Check for correlations between individual features:

# In[34]:


train.corr()


# In[35]:


test.corr()


# In[3]:


r2 = pd.concat([train.drop(['target', 'ID_code'], axis=1), test.drop('ID_code', axis=1)]).corr()**2
r2 = np.tril(r2, k=-1)  # remove upper triangle and diagonal
r2[r2 == 0] = np.nan # replace 0 with nan


# In[4]:


f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(np.sqrt(r2), annot=False,cmap='viridis', ax=ax)


# Explained absolute variation between individual features is small (< 1%). All features seem to be highly independent from each other.
# 
# What about correlations between features and target?

# In[5]:


target_r2 = train.drop(['ID_code', 'target'], axis=1).corrwith(train.target).agg('square')

f, ax = plt.subplots(figsize=(25,5))
target_r2.agg('sqrt').plot.bar(ax=ax)


# ## Data Visualization

# ### Density plots of features
# Let's show now the density plot of variables in train dataset.
# 
# We represent with different colors the distribution for values with target value 0 and 1.

# In[17]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[20]:


get_ipython().run_cell_magic('time', '', "t0 = train.loc[train['target'] == 0]\nt1 = train.loc[train['target'] == 1]\nfeatures = train.columns.values[2:102]\nplot_feature_distribution(t0, t1, '0', '1', features)")


# In[21]:


get_ipython().run_cell_magic('time', '', "\nfeatures = train.columns.values[102:202]\nplot_feature_distribution(t0, t1, '0', '1', features)")


# In[23]:


get_ipython().run_cell_magic('time', '', "\nfeatures = train.columns.values[2:102]\nplot_feature_distribution(train, test, 'train', 'test', features)")


# In[24]:


get_ipython().run_cell_magic('time', '', "\nfeatures = train.columns.values[102:202]\nplot_feature_distribution(train, test, 'train', 'test', features)")


# ### Distribution of mean and std
# Let's check the distribution of the mean values per row in the train and test set.

# In[25]:


plt.figure(figsize=(16,6))
features = train.columns.values[(2:202)]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[26]:


get_ipython().run_cell_magic('time', '', '\nplt.figure(figsize=(16,6))\nplt.title("Distribution of mean values per column in the train and test set")\nsns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label=\'train\')\nsns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label=\'test\')\nplt.legend()\nplt.show()')


# In[28]:


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[29]:


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[30]:


plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis per column in the train and test set")
sns.distplot(train[features].kurtosis(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[31]:


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend()
plt.show()


# ## Predictive Analysis Modeling
# ---

# In[1]:


get_ipython().run_cell_magic('html', '', '<iframe src="https://lightgbm.readthedocs.io/en/latest/" width="1000" height="300"></iframe>')


# In[56]:


features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']


# In[57]:


@jit
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)
        
        
    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# ## Parameters Tuning Guide

# In[5]:


get_ipython().run_cell_magic('html', '', '<iframe src="https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html" width="1000" height="300"></iframe>')


# ## Parameter Tuning
# ---

# In[6]:


get_ipython().run_cell_magic('html', '', '<iframe src="https://buildmedia.readthedocs.org/media/pdf/lightgbm/latest/lightgbm.pdf" width="1000" height="300"></iframe>')


# In[58]:


param = {
    'bagging_freq': 5,  #handling overfitting
    'bagging_fraction': 0.335,  #handling overfitting - adding some noise
    'bagging_seed': 7,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,#handling overfitting
    'learning_rate': 0.08,
    'reg_alpha': 1.728910519108444,
    'reg_lambda': 4.9847051755586085,
    'subsample': 0.81,
    'min_gain_to_split': 0.01077313523861969,
    'min_child_weight': 19.428902804238373,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}


# In[59]:


num_folds = 11
features = [c for c in train.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=44000)
oof = np.zeros(len(train))
getVal = np.zeros(len(train))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()


# # Logistic Regression Model:-

# In[ ]:


#Training Data:
X=df_train.drop(['ID_code','target'],axis=1)
Y=df_train['target']

#Stratified KFold Cross Validator:-
skf=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
for train_index, valid_index in skf.split(X,Y): 
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index] 
    y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
    
print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# Shape of X_train : (160001, 200)
# 
# Shape of X_valid : (39999, 200)
# 
# Shape of y_train : (160001,)
# 
# Shape of y_valid : (39999,)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lr_model=LogisticRegression(random_state=42)\n#fitting the model-\nlr_model.fit(X_train,y_train)')


# In[ ]:


#Accuracy of model-
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of lr_model :',lr_score)


# Accuracy of lr_model : 0.9148942819107381

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Cross validation prediction of lr_model-\ncv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)\n#Cross validation score-\ncv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)\nprint('cross val score :',np.average(cv_score))")


# cross val score : 0.9116728528566072
# 
# Wall time: 13min 24s

# # Confusion Matrix:-

# In[ ]:


#Confusion matrix:-
cm=confusion_matrix(y_valid,cv_predict)
cm=pd.crosstab(y_valid,cv_predict)
cm


#       col_0  0	1
# 
#     target
# 
#     0	35436	544
#     1	2989	1030

# In[ ]:


#ROC_AUC SCORE:-
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC Score:',roc_score)


# In[ ]:


#ROC_AUC_Curve:-
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# ![image.png](attachment:image.png)

# AUC: 0.620581573269051

# **Observation:-** On comparing roc_auc_score and model accuracy, model is not performing well on imbalanced data.

# In[ ]:


#Classification Report:-
scores=classification_report(y_smote_v,cv_pred)
print(scores)

              precision    recall  f1-score   support

           0       0.81      0.79      0.80     35980
           1       0.79      0.81      0.80     35980

    accuracy                           0.80     71960
   macro avg       0.80      0.80      0.80     71960
weighted avg       0.80      0.80      0.80     71960
# Observation:- As we see that f1 score is high for the customers who will not make a transaction, as well as who will make a transaction.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Predicting the model-\nX_test=df_test.drop(['ID_code'],axis=1)\nsmote_pred=smote.predict(X_test)\nprint(smote_pred)")


# Observation:- We can observe that the smote model is performing well on imbalance data as compare to logistic regression.

# **Light GBM:-** It is a gradient boosting framework that uses tree based learning algorithm.

# In[ ]:


#Training lgbm model:-
num_rounds=10000
lgbm= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],verbose_eval=1000,early_stopping_rounds = 5000)
lgbm


# In[60]:


get_ipython().run_cell_magic('time', '', '\nprint(\'Light GBM Model\')\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):\n    \n    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]\n    X_valid, y_valid = train.iloc[val_idx][features], target.iloc[val_idx]\n    \n    \n    X_tr, y_tr = augment(X_train.values, y_train.values)\n    X_tr = pd.DataFrame(X_tr)\n    \n    \n    print("Fold idx:{}".format(fold_ + 1))\n    trn_data = lgb.Dataset(X_tr, label=y_tr)\n    val_data = lgb.Dataset(X_valid, label=y_valid)\n    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])\n    \n    \n    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, \n                    early_stopping_rounds = 3000)\n    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    getVal[val_idx]+= clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits\n    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits\n        \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)')


# In[61]:


print("\n >> CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


# # Conclusion:- 
# We have tried with diff model like Logistic regression,smote & lightgbm. But we observed that light gbm is performing well on imbalanced data as compare to other models based on the roc_auc scores.

# In[11]:



cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[63]:


submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = predictions
submission.to_csv("submission_final.csv", index=False)

