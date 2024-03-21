#!/usr/bin/env python
# coding: utf-8

# ### Importing Tools and Libraries

# In[4]:


#EDA and Visualization Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Modelling Tools
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#HyperParameter Tuning and Cross Validation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay 


# In[11]:


import sklearn
sklearn.__version__ # will print out the version number


# ## Load Data

# In[12]:


df = pd.read_csv('D:\OneDrive\Desktop\heart-disease.csv')


# In[6]:


df


# In[13]:


df.shape


# ## Exploratory Data Analysis

# In[14]:


df.head()


# In[15]:


df.tail()


# In[16]:


df['target'].value_counts()


# In[17]:


df['target'].value_counts().plot(kind = 'bar', color = ['salmon','lightblue'])


# In[19]:


df.info()


# In[22]:


df.isna().sum()


# In[24]:


df.describe(include = 'all')


# In[27]:


df.sex.value_counts().plot(kind = 'bar',color = ['blue','green'])


# In[28]:


#Checking the heart disease frequency for sex

pd.crosstab(df.target,df.sex)


# In[36]:


pd.crosstab(df.target,df.sex).plot(kind = 'bar', figsize = (10,6))
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Amount')
plt.legend(['Female','Male'])
plt.xticks(rotation = 0)


# In[41]:


plt.figure(figsize = (10,6))

#Scatter with positive examples

plt.scatter(df.age[df.target == 1],df.thalach[df.target == 1])

#Scatter with negative examples

plt.scatter(df.age[df.target == 0],df.thalach[df.target == 0])

#Making the graph readable
plt.title('Heart Disease plot with age and heart rate')
plt.xlabel('Age')
plt.ylabel('Max Heart rate')
plt.legend(['Disease','No Disease'])


# In[42]:


#Distribution of age

df.age.plot.hist()


# In[45]:


pd.crosstab(df.cp,df.target)


# In[52]:


pd.crosstab(df.cp,df.target).plot(kind = 'bar' , figsize = (10,6))
plt.title('Heart Disease with different types of chest pains')
plt.xlabel(['0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain,3 = Asymptomatic'])
plt.ylabel('Amount')
plt.legend(['No Disease','Disease'])
plt.xticks(rotation = 0)


# In[53]:


df.corr()


# In[54]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(corr_matrix, annot = True, linewidth = 0.5, fmt = '.2f', cmap = 'YlGnBu')


# ## Modelling

# In[56]:


df.head()


# In[13]:


X = df.drop('target',axis = 1)
y = df['target']


# In[59]:


y


# In[14]:


np.random.seed(42)

X_train,X_Test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[61]:


X_train


# In[2]:


from sklearn.linear_model import LogisticRegression


# In[7]:


models= {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(), 'Random Forest' : RandomForestClassifier()}


# In[18]:


def fit_and_score(models,X_train,X_Test,y_train,y_test):
    
    np.random.seed(42)
    
    model_scores = {}
    
    for name,model in models.items():
        
        model.fit(X_train,y_train)
        
        model_scores[name] = model.score(X_Test,y_test)
        
    return model_scores
    
    


# In[19]:


model_scores = fit_and_score(models = models,
                             X_train = X_train,
                             X_Test = X_Test,
                             y_train = y_train,
                             y_test = y_test)

model_scores


# In[22]:


model_compare = pd.DataFrame(model_scores, index = ['Accuracy'])
model_compare.T.plot.bar()


# ### Hyper Parameter Tuning

# In[25]:


train_scores = []
test_scores = []

#HyperParameter Tuning for KNN

neighbors = range(1,21)

knn = KNeighborsClassifier()

for i in neighbors:
    
    knn.set_params(n_neighbors=i)
    
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    
    test_scores.append(knn.score(X_Test,y_test))


# In[26]:


train_scores


# In[27]:


test_scores


# In[33]:


plt.plot(neighbors,train_scores, label = 'Train Scores')
plt.plot(neighbors,test_scores,label = 'Test Scores')
plt.xticks(np.arange(1,21,1))
plt.xlabel('Neighbors')
plt.ylabel('Model Scores')
plt.legend()
print(f"Maximum score on the test data: {max(test_scores)*100: 2f}%")


# ### Hyperparameter Tuning with RandomizedSearchCV

# In[39]:


#Create a Hyperparameter Grid for Logistic Regression

log_reg_grid = {'C': np.logspace(-4,4,20), 
                'solver': ['liblinear']}

#Create a Hyperparameter Grid for Random Forest Classifier

rf_grid = {'n_estimators' : np.arange(10,1000,50),
          'max_depth' : [None,3,5,10],
          'min_samples_split' : np.arange(2,20,2),
          'min_samples_leaf' : np.arange(1,20,2)}


# In[35]:


#Tune Logistic Regression

np.random.seed(42)

#Setup Hyperparameter search for Logistic Regression

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv = 5,
                                n_iter= 20,
                                verbose= True)

rs_log_reg.fit(X_train,y_train)


# In[36]:


rs_log_reg.best_params_


# In[37]:


rs_log_reg.score(X_Test,y_test)


# In[40]:


# Tuning Randomforest Classifier

np.random.seed(42)

#Setup Random Hyperparameter search

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions= rf_grid,
                           cv= 5,
                           n_iter= 20,
                           verbose= True
                           )

#Fit random HyperParameter search model for random forest classifier

rs_rf.fit(X_train,y_train)


# In[41]:


rs_rf.best_params_


# In[42]:


rs_rf.score(X_Test,y_test)


# In[44]:


#Hyperparameter tuning using gridsearchCV

log_reg_grid = {'C': np.logspace(-4,4,30), 
                'solver': ['liblinear']}

gs_log_reg = GridSearchCV(LogisticRegression(), 
                          param_grid= log_reg_grid,
                          cv = 5,
                          verbose = True)

gs_log_reg.fit(X_train,y_train)


# In[45]:


gs_log_reg.best_params_


# In[46]:


gs_log_reg.score(X_Test,y_test)


# ### Evaluation Metrics

# In[54]:


y_preds = gs_log_reg.predict(X_Test)


# In[56]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Assuming gs_log_reg is your trained model and X_test, y_test are your test data

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, gs_log_reg.predict_proba(X_Test)[:, 1])

# Create a RocCurveDisplay object and plot the ROC curve
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()


# In[57]:


print(confusion_matrix(y_test,y_preds))


# In[59]:


sns.set(font_scale = 1.5)

def plot_conf_matrix(y_test,y_preds):
    
    fig,ax = plt.subplots(figsize = (3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                    annot = True,
                    cbar = False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    
plot_conf_matrix(y_test,y_preds)


# In[60]:


print(classification_report(y_test,y_preds))


# ### Evaluating Classification Report using cross validation

# In[61]:


# Check best hyperparameters

gs_log_reg.best_params_


# In[63]:


#Create a new classifier using best parameters that we found

clf = LogisticRegression(C=0.20433597178569418,
                         solver = 'liblinear')


# In[64]:


#Cross-Validated Accuracy

cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring='accuracy')

cv_acc


# In[65]:


np.mean(cv_acc)


# In[66]:


#Cross-Validated precision

cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring='precision')

cv_precision = np.mean(cv_precision)

cv_precision


# In[69]:


#Cross-Validated recall

cv_recall= cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring='recall')

cv_recall = np.mean(cv_recall)

cv_recall


# In[68]:


#Cross-Validated f1

cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv = 5,
                         scoring='f1')

cv_f1 = np.mean(cv_f1)

cv_f1


# In[75]:


#Visualize cross validated scored

cv_metrics = pd.DataFrame({'Accuracy' : cv_acc,
                           'Precision': cv_precision,
                           'Recall': cv_recall,
                           'F1-Scores': cv_f1})

cv_metrics.T.plot.bar(title = 'Cross-Validated Metrics',
                     legend = False)


# ### Feature Importance

# In[76]:


gs_log_reg.best_params_


# In[77]:


clf = LogisticRegression(C =0.20433597178569418, solver = 'liblinear' )

clf.fit(X_train,y_train)


# In[78]:


#Check coeffs
clf.coef_


# In[79]:


#Match coefs of features to columns

feature_dict = dict(zip(df.columns,list(clf.coef_[0])))

feature_dict


# In[81]:


feature_df = pd.DataFrame(feature_dict,index = [0])

feature_df.T.plot.bar(title = 'Feature Importance Graph',legend = False)

