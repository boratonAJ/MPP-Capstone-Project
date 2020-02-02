
# coding: utf-8

# In[16]:


#Fundamental librarys to math and stats process
import numpy as np
import numpy.random as nr
import scipy.stats as ss
import math
#data prepared
import pandas as pd

#ML preprocessi
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn.pipeline import make_pipeline
from sklearn import feature_selection as fs


from sklearn.preprocessing import RobustScaler,Normalizer, MinMaxScaler,FunctionTransformer, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge

# Multiclass neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

#RandomForest
from sklearn.ensemble import RandomForestClassifier

#DecisionTree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# XGBoost
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# ML algorithms models
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

# ML Evaluations
import sklearn.metrics as sklm
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split

#Ploting
import matplotlib.pyplot as plt
import seaborn as sns


#get_ipython().magic('matplotlib inline')
plt.ion

import warnings
warnings.filterwarnings('ignore')


# In[17]:


df = (pd.read_csv('../Data/df_enc_2.csv'))


# In[18]:


df.shape


# In[19]:


df.head()


# In[20]:


x= df.drop(['Unnamed: 0', 'accepted'], axis=1)
y= df['accepted']


# In[21]:


x.head()


# In[22]:


y.head()


# Select features according to the k highest scores.
# some sort of normalized values such as z-scores and therefore don't want to do any more normalization,then you should consider using the ANOVA (f_classif) scoring function for your feature selection. If you are using z-score normalization or some other normalization that uses negatives (maybe your data falls between -1 and +1),you could just use f_classif scoring function which doesn't require only positive numbers.

# In[23]:


bestfeatures = fs.SelectKBest(score_func=fs.f_classif, k=20)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(30,'Score'))  #print 10 best features


# In[24]:


int_cols = ['loan_purpose','False_Col','True_Col',
           'state_code','property_type','msa_md',
            'county_code','loan_amount','applicant_race',
           'applicant_sex']


# In[25]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(30).plot(kind='barh')
#plt.show()


# In[26]:


x=x[int_cols]
x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)


# ## ## Train models selected

# ### Split Data
#
# Random state (Pseudo-random number) in Scikit learn train_test_split splits arrays or matrices into random train and test subsets. That means that everytime you run it without specifying random_state, you will get a different result, this is expected behavior.It changes. On the other hand if you use random_state=some_number, then you can guarantee that the output of Run 1 will be equal to the output of Run 2, i.e. your split will be always the same. It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing is that everytime you use 42, you will always get the same output the first time you make the split. This is useful if you want reproducible results, for example in the documentation, so that everybody can consistently see the same numbers when they run the examples. In practice I would say, you should set the random_state to some fixed number while you test stuff, but then remove it in production if you really need a random (and not a fixed) split.Regarding your second question, a pseudo-random number generator is a number generator that generates almost truly random numbers. Why they are not truly random is out of the scope of this question and probably won't matter in your case, you can take a look here form more details.Pseudorandom number generator

# In[27]:


# splt train test
nr.seed(9988)
x_train, x_test, y_train, y_test = train_test_split(x,y)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=54321)


# In[28]:


x_train.shape, y_train.shape


# In[29]:


x_test.shape, y_test.shape


# ### Linear Model

# In[30]:


lasso = make_pipeline(RobustScaler(), Lasso())
ENet = make_pipeline(RobustScaler(),ElasticNet())
KRR = KernelRidge(kernel='polynomial', degree=2)


# In[ ]:


lasso.fit(x_train,y_train)
ENet.fit(x_train,y_train)
KRR.fit(x_train,y_train)


# In[ ]:


# cross validation
Results_Lasso = cross_validate(lasso,x,y,scoring="r2",cv=5)
Results_Enet = cross_validate(ENet,x,y,scoring="r2",cv=5)
Results_KRR = cross_validate(KRR,x,y,scoring="r2",cv=5)


# In[ ]:


Lasso_test_scores = Results_Lasso['test_score']
Lasso_train_scores = Results_Lasso['train_score']
Enet_test_scores = Results_Enet['test_score']
Enet_train_scores = Results_Enet['train_score']
KRR_test_scores = Results_KRR['test_score']
KRR_train_scores = Results_KRR['train_score']

print(np.mean(Lasso_train_scores))
print(np.mean(Lasso_test_scores))
print(np.mean(Enet_train_scores))
print(np.mean(Enet_test_scores))
print(np.mean(KRR_train_scores))
print(np.mean(KRR_test_scores))


# In[ ]:


estimator = make_pipeline(RobustScaler(), Lasso())
estimator.fit(x,y)


# ## Make Prediction and Output for Scoring
#
# Before we predict and export our score we need to apply the same changes to our test set as our training set.

# In[ ]:


test_values=pd.read_csv('Data/test_values.csv')
test_values= test_values.fillna(test_values.mean())
test_values=np.array(test_values[int_cols])


# In[ ]:


L_prediccion=pd.DataFrame(estimator.predict(test_values),columns=['accepted'])
L_prediccion.index.names=['row_id']
L_prediccion['accepted']= prediccion['accepted'].astype(np.int64)
L_prediccion.head()


# In[ ]:


prediccion.to_csv('Data/submission.csv')


# ### Score Models
