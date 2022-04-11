#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PREDICT CUSTOMER CHURN IN A BANK


# In[2]:


get_ipython().system('pip install catboost')


# In[4]:


get_ipython().system('pip install xgboost')


# In[3]:


#load necccessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

import os, sys
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#load dataset
churn = pd.read_csv('banking_churn.csv')
data = churn.copy()


# In[ ]:


#check size of the observation and variable
data.shape


# In[ ]:


#load the first 5 data in the dataset
data.head()


# In[ ]:


#load columns in the datasets
data.columns


# In[ ]:


#load information about the dataset
data.info()


# In[ ]:


#checking for missing data
data.isnull().any()


# In[ ]:


data.drop(["RowNumber","CustomerId","Surname"], axis = 1 , inplace = True)
data.columns


# In[ ]:


#statistics of the data
stats = data.describe()
stats


# In[ ]:


# Analyzing target variable
plt.figure(figsize = (15,8))
sns.countplot('Exited', data = data)


# In[ ]:


# Analyzing how categorical data relates with the target variable (exited)
cat_data = data[['Gender', 'Tenure','Geography', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']]

def categorical(var):
    print(data[var].value_counts())
    
    plt.figure(figsize = (15,8))
    sns.countplot(x = var, data = data, hue = 'Exited')
    plt.show()
    
for i in cat_data:
    categorical(i)


# In[ ]:


# Analyzing numerical data
Num_data = data[[ 'CreditScore', 'Age', 'Balance', 'EstimatedSalary' ]]
def numerical(var):

    plt.hist(data[var], bins = 20, color = "brown")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.title("{} variable distribution".format(var))
    plt.show()
    
for i in Num_data:
    numerical(i)


# In[ ]:


## Visualizing outliers 
listOrder = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']


def viz_outliers(var):

    sns.boxplot(data[var])
    plt.show()
    
for i in listOrder:
    viz_outliers(i)


# In[ ]:


# Observations
# There is presence of outliers in CreditScore, Age, NumOfProducts
outliers = ['Age','CreditScore','NumOfProducts']

# create a function to remove the outliers
def outlier_removal(data,column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    point_low = q1 - 1.5 * iqr
    point_high = q3 + 1.5 * iqr
    cleaned_data = data.loc[(data[column] >  point_low) & (data[column] <  point_high)]
    return cleaned_data

# clean the dataset by removing outliers
data_cleaned = outlier_removal(outlier_removal(outlier_removal(data,'Age'),'CreditScore'),'NumOfProducts')

print(data.shape)
print(data_cleaned.shape)


# In[ ]:


# Correlation Matrix 

plt.figure(figsize = (15,8))
list_corr = ['CreditScore' ,'Age' ,'Tenure' ,'Balance' ,'NumOfProducts' ,'EstimatedSalary' ,'Exited']
sns.heatmap(data_cleaned[list_corr].corr(), annot = True, linecolor = "green", lw = 0.5, fmt= '.2f') 


# In[ ]:


# # Observation
# Age has the strongest relation with Exited (0.36).
#   As the age of the customer increases, the rate of losing the customer increases. (Positive strong relationship)
# Exited and Balance variable have a relatively strong relationship (0.11).
# Exited and the variable NumOfProducts have a moderately strong relationship (-0.11).
#  They have a strong negative relationship.


# In[ ]:


# Analyzing how numerical variable relates with the target variable (exited)
# AGE AND EXIT

plt.figure(figsize = (15,8))
sns.lineplot(x = "Age", y = "Exited", data = data_cleaned)


# In[ ]:


data_cleaned.groupby(data_cleaned["Exited"])["Age"].mean()


# In[ ]:


# Observation
# As the age of the customer increases, the customer losing rate increases.
# Average age of customers who did not leave the bank is 36
# Average age of customers leaving the bank is 43


# In[ ]:


# Feature Engineering
# since geography is a categorical data lets one-hot encode it by using pd.get_dummies
data_cleaned = pd.get_dummies(data_cleaned, columns = ['Geography'])

# since gender is a categorical data lets label encode it as female = 1 and male = 0
def func(data_cleaned):
    d =[]
    for m in data_cleaned:
        if m =='Female':
            d.append(1)
        else:
            d.append(0)
    return d

data_cleaned['Gender'] = func(data_cleaned['Gender'])


# In[ ]:


data_cleaned.info()


# In[ ]:





# In[ ]:


# Modelling.

x = data_cleaned.drop('Exited', axis = 1) 
y = data_cleaned['Exited']

#splitting data into test and train set
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 42
                                                   )


# In[ ]:





# In[ ]:


# LogisticRegression

Lr = LogisticRegression() # algorithm instantiation
Lr.fit(x_train, y_train) # model learning

# make your predictions on the test data
pred = Lr.predict(x_test)

print(classification_report(y_test, pred, digits=2))

# evaluate the test data using accuracy score
print("Accuracy score of Logistic Regression model: ", accuracy_score(y_test, pred))


# In[ ]:


#Perform feature scaling (standardization) using standardscalar()
sc = StandardScaler()
xstandard_train = sc.fit_transform(x_train)
xstandard_test = sc.transform (x_test)


# In[ ]:


Lr_s = LogisticRegression() # algorithm instantiation
Lr_s.fit(xstandard_train, y_train)

pred = Lr_s.predict(xstandard_test)

print(classification_report(y_test, pred, digits=2))
Lr_score = accuracy_score(y_test, pred)
print("Accuracy score of Standardised Logistic Regression model: ", accuracy_score(y_test, pred))


# In[ ]:





# In[ ]:


# SVC

svc = SVC(probability = True) # algorithm instantiation
svc.fit(xstandard_train, y_train)

pred = svc.predict(xstandard_test)

print(classification_report(y_test, pred, digits=2))
svc_score = accuracy_score(y_test, pred)
print("Accuracy score of SVC model: ", accuracy_score(y_test, pred))


# In[ ]:


#KNN

knn = KNeighborsClassifier(n_neighbors = 5) # algorithm instantiation
knn.fit(xstandard_train, y_train)

pred = knn.predict(xstandard_test)

print(classification_report(y_test, pred, digits=2))
knn_score = accuracy_score(y_test, pred)
print("Accuracy score of KNN model: ", accuracy_score(y_test, pred))


# In[ ]:





# In[ ]:


# RandomForestClassifier

rand = RandomForestClassifier(random_state = 42)

rand.fit(x_train, y_train)
pred = rand.predict(x_test)

print(classification_report(y_test, pred, digits=2))
rand_score = accuracy_score(y_test, pred)
print("Accuracy score of Random Forest model: ", accuracy_score(y_test, pred))


# In[ ]:


rand = RandomForestClassifier(random_state = 42, max_depth = 10, n_estimators = 1000)

rand.fit(x, y)

scoreRand = cross_val_score(rand, x, y, cv=5, scoring='accuracy')
print ('The mean value of cross val score is {}'.format(scoreRand.mean()))


# In[ ]:





# In[ ]:


# XGBClassifier

xgb =XGBClassifier(learning_rate=0.01,n_estimators=200, 
                            max_depth=5, eval_metric="logloss")

xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)

print(classification_report(y_test, pred, digits=2))
xgb_score = accuracy_score(y_test, pred)
print("Accuracy score of XGB model: ", accuracy_score(y_test, pred))


# In[ ]:


xgb =XGBClassifier()
xgb.fit(x, y)

scoreXGB = cross_val_score(xgb, x, y, cv=10, scoring='accuracy')
print ('The mean value of cross val score is {}'.format(scoreXGB.mean()))


# In[ ]:





# In[ ]:


# GradientBoostingClassifier

model_grb = GradientBoostingClassifier()

model_grb.fit(x_train,y_train)

pred = model_grb.predict(x_test)

print(classification_report(y_test, pred, digits=2))
grb_score = accuracy_score(y_test, pred)
print("Accuracy score of Gradient Boost model: ", accuracy_score(y_test, pred))


# In[ ]:


model_grb = GradientBoostingClassifier()
model_grb.fit(x, y)

scoreGRB = cross_val_score(model_grb, x, y, cv=10, scoring='accuracy')
print ('The mean value of cross val score is {}'.format(scoreGRB.mean()))


# In[ ]:





# In[ ]:


# LGBM_Model

lgbm_model = LGBMClassifier(silent = 0, learning_rate = 0.09, max_delta_step = 2, n_estimators = 100, boosting_type = 'gbdt',
                            max_depth = 10, eval_metric = "logloss", gamma = 3, base_score = 0.5)

lgbm_model.fit(x_train, y_train)
y_pred = lgbm_model.predict(x_test)
print(classification_report(y_test, y_pred, digits=2))
lgbm_score = accuracy_score(y_test, y_pred)
print("Accuracy score of tuned LightGBM model: ", accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:


# CatBoostClassifier

#Instantiate CatBoostClassifier
catboost = CatBoostClassifier()

#create the grid
grid = {'max_depth': [3,4,5],'n_estimators':[100, 200, 300]}

#Instantiate GridSearchCV
gscv = GridSearchCV (estimator = catboost, param_grid = grid, scoring ='accuracy', cv = 5)

#fit the model
gscv.fit(x_train,y_train)

#returns the estimator with the best performance
print(gscv.best_estimator_)

#returns the best score
print(gscv.best_score_)

#returns the best parameters
print(gscv.best_params_)


# In[ ]:


cat = CatBoostClassifier(max_depth = 4, n_estimators = 200 , verbose=0)

cat.fit(x_train, y_train)
pred = cat.predict(x_test)

print(classification_report(y_test, pred, digits=2))
cat_score = accuracy_score(y_test, pred)
print("Accuracy score of CatBoost model: ", accuracy_score(y_test, pred))


# In[ ]:


cat = CatBoostClassifier(max_depth=4, n_estimators=1000, verbose=0)

cat_tuned = cat.fit(x, y)
scoreCat = cross_val_score(cat, x, y, cv=5, scoring='accuracy')
print ('The mean value of cross val score is {}'.format(scoreCat.mean()))


# In[ ]:


# EMSEMBLE
model = VotingClassifier(estimators=[('catboost', cat), ('LGBM', lgbm_model), ('GradientBoost', model_grb), ('randomforest', rand)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[ ]:


model_data =[['LightGBM Classifier', lgbm_score],
             ['Random Forest Classifier', rand_score],
             ['Catboost Classifier', cat_score], 
             ['XGB Classifier', xgb_score],
             ['Gradient Boost Classifier', grb_score],
             ['SVM Classifier', svc_score],
             ['Logistic Regression', Lr_score],
             ['KNN Classifier', knn_score]] 

indexes = [1,2,3,4,5,6,7,8]
columns_name = ['MODEL', 'ACCURACY_SCORE']
fmw = pd.DataFrame(data = model_data,index = indexes, columns = columns_name )
print(fmw)


# In[ ]:


# Evaluate model using Area under Curve to evaluate best performed model

plot_roc_curve(cat, x_test, y_test) 
plot_roc_curve(rand, x_test, y_test) 
plot_roc_curve(model_grb, x_test, y_test)
plot_roc_curve(svc, x_test, y_test) 
plot_roc_curve(knn, x_test, y_test) 
plot_roc_curve(Lr_s, x_test, y_test)
plot_roc_curve(xgb, x_test, y_test) 
plot_roc_curve(lgbm_model, x_test, y_test) 
plt.show()


# In[ ]:


# Summary:
# The best model amonst the ones implemented is Random Forests with an accuracy of 86.2%


# In[ ]:


#feature importance of random forest model i.e the most importance predictive feature (variables) in the model performance
feature_index = data_cleaned.loc[:, x.columns ]

feature_importance = pd.Series(rand.feature_importances_, 
                               index=feature_index.columns).sort_values(ascending=False)
                               
sns.barplot(x = feature_importance, y = feature_importance.index, color='brown')
plt.xlabel('Variable Importance Scores')
plt.ylabel('Variables')
plt.title('Random Forest Feature Importance')
plt.show()


# In[ ]:


# Deployment
# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(rand, pickle_out) 
pickle_out.close()


# In[ ]:


get_ipython().system('pip install -q pyngrok')


# In[ ]:


get_ipython().system('pip install -q streamlit')


# In[ ]:


get_ipython().system('pip install -q streamlit_ace')


# In[ ]:


data_cleaned.head()


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', ' \nimport pickle\nimport streamlit as st\n \n# loading the trained model\npickle_in = open(\'classifier.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n \n@st.cache()\n  \n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain):   \n \n    # Pre-processing user input    \n    if Gender == "Male":\n        Gender = 0\n    else:\n        Gender = 1\n \n    if HasCrCard == "No":\n        HasCrCard = 0\n    else:\n        HasCrCard = 1\n \n    if IsActiveMember == "No":\n        IsActiveMember = 0\n    else:\n        IsActiveMember = 1  \n\n    if Geography_France == "Yes":\n        Geography_France = 1\n    else:\n        Geography_France = 0\n\n    if Geography_Spain == "Yes":\n        Geography_Spain = 1\n    else:\n        Geography_Spain = 0\n    \n    if Geography_Germany == "Yes":\n        Geography_Germany = 1\n    else:\n        Geography_Germany = 0\n\n \n    # Making predictions \n    prediction = classifier.predict( \n        [[CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain]])\n     \n    if prediction == 0:\n        pred = \'Stay!\'\n    else:\n        pred = \'Leave!\'\n    return pred\n\n\n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:brown;padding:13px"> \n    <h1 style ="color:gray1;text-align:center;">Streamlit Bank Customer Churn Prediction MLApp</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n      \n    # following lines create boxes in which user can enter data required to make prediction \n    Gender = st.selectbox("Customer\'s Gender",("Male","Female"))\n    Age = st.number_input("Customer\'s Age")\n    NumOfProducts = st.selectbox("Total Number of Bank Product The Customer Uses", ("1","2","3","4"))\n    Tenure = st.selectbox("Number of Years The Customer Has Been a Client", ("0","1","2","3","4","5","6","7","8","9","10"))\n    HasCrCard = st.selectbox(\'Does The Customer has a Credit Card?\',("Yes","No"))\n    IsActiveMember = st.selectbox(\'Is The Customer an Active Member?\',("Yes","No"))\n    EstimatedSalary = st.number_input("Estimated Salary of Customer") \n    Balance = st.number_input("Customer\'s Account Balance")\n    CreditScore = st.number_input("Customer\'s Credit Score")\n    Geography_France = st.selectbox(\'Is the Customer From France?\',("Yes","No"))\n    Geography_Spain = st.selectbox(\'Is the Customer From Spain?\',("Yes","No"))\n    Geography_Germany = st.selectbox(\'Is the Customer From Germany?\',("Yes","No"))\n\n    result =""\n      \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain) \n        st.success(\'The Customer will {}\'.format(result))\n        #print(LoanAmount)\n     \nif __name__==\'__main__\': \n    main()')


# In[ ]:


get_ipython().system('streamlit run app.py &>/dev/null&')


# In[ ]:


from pyngrok import ngrok
 
public_url = ngrok.connect('8888')
public_url


# In[ ]:




