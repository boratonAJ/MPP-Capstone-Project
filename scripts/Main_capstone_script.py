#!/usr/bin/python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/

# Any results you write to the current directory are saved as output.
#https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# Python script for confusion matrix creation.
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Fundamental librarys to math and stats process
import numpy.random as nr
import scipy.stats as ss
import math
import fancyimpute
from collections import Counter
import time
import pickle
import xgboost as xgb
import seaborn as sns
import category_encoders as ce
from mlens.visualization import corrmat
from xgboost import XGBClassifier, XGBRegressor
from keras import models, layers, optimizers
#ML preprocessing and ML algorithms models
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as ms
from sklearn.pipeline import make_pipeline
from sklearn import feature_selection as fs
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler,Normalizer, MinMaxScaler,FunctionTransformer, PolynomialFeatures, Imputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
# ML Evaluations
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import Imputer, RobustScaler, FunctionTransformer, StandardScaler
# ML sklearn  algorithms models
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sklm
from sklearn.metrics import (roc_auc_score, confusion_matrix,classification_report,
                             accuracy_score, roc_curve, auc,
                             precision_recall_curve, f1_score, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsemble
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
from matplotlib import pyplot
#get_ipython().magic('matplotlib inline')
plt.ion
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
import warnings
warnings.filterwarnings('ignore')
from matplotlib import*

# Input data files are available in the "../Data/" directory.
DATA_File = os.listdir('../Data/')[0]
print(os.listdir("../Data/"))
nr.seed(3456)
rng = np.random.RandomState(31337)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def data_transform():
    df = (pd.read_csv('../Data/df_enc_3.csv'))
    return df

# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
df = data_transform()
df= df.drop(['Unnamed: 0'], axis=1)
df = clean_dataset(df)
X = df.drop('accepted', axis = 1)  #df[df.columns[:-1].tolist()]
cate = X.columns.tolist() #categorical[:-1]
for col in cate:
    X[col] = X[col].astype('category')
categorical_features_pos = column_index(X,cate)


def scale_numeric(data, numeric_columns, scaler):
    for col in numeric_columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
    return data


def data_pre_processing():
    df = data_transform()

    df= df.drop(['Unnamed: 0'], axis=1)
    #Create binary features to check if the example is has missing values for all features that have missing values
    for feature in df.columns:
        if np.any(np.isnan(df[feature])):
            df["is_" + feature + "_missing"] = np.isnan(df[feature]) * 1
    # split into X and y
    X = df.drop('accepted', axis = 1)
    # Convert to numpy array
    X=np.array(X)
    # Labels are the values we want to predict
    y = df['accepted']
    # Use numpy to convert to arrays
    y=np.array(y)
    # instantiate an encoder - here we use Binary()
    return X, y

def data_processor():
    X, y = data_pre_processing()
    print("Train Test Split ratio is 0.2")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=54321)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, shuffle=True, random_state=54321)
    print(f"Original data shapes: {X_train.shape, X_test.shape}")
    print(f"Original data shapes: {y_train.shape, y_test.shape}")
    return X_train, X_test, y_train, y_test


def best_festure():
    df = data_transform()
    X, y = data_pre_processing()
    bestfeatures = fs.SelectKBest(score_func=fs.f_classif, k=20)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(df.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(14,'Score'))  #print 10 best features



def plot_best_feature(df,X_train,y_train):
    X, y = data_pre_processing()
    X_train, X_test, y_train, y_test = data_processor()
    # fit RF to plot feature importances
    rf_clf.fit(RobustScaler().fit_transform(Imputer(strategy="median").fit_transform(X_train)), y_train)
    # Plot features importance
    importances = rf_clf.feature_importances_
    indices = np.argsort(rf_clf.feature_importances_)[::-1]
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, 46), importances[indices], align="center")
    plt.xticks(range(1, 46), df.columns[df.columns != "accepted"][indices], rotation=90)
    plt.title("Feature Importance", {"fontsize": 16});
    plt.show()


def getbaseline():
    """
    We first fit a decision tree with default parameters to get a baseline idea of the performance
    """
    X_train, X_test, y_train, y_test = data_processor()
    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(X_train, y_train)
    y_pred = decisiontree.predict(X_test)
    y_pred_proba = decisiontree.predict_proba(X_test)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Decision Tree Accuracy Score: ", decisiontree.score(X_test, y_test))
    print("Decision Tree F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("Decision Tree AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
    print("Decision Tree ROC_AUC Score is: {}".format(roc_auc))


def plot_baseline():
    #https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
    X_train, X_test, y_train, y_test = data_processor()
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
       decisiontree = DecisionTreeClassifier(max_depth=max_depth)
       decisiontree.fit(X_train, y_train)
       train_pred = decisiontree.predict(X_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous train results
       train_results.append(roc_auc)
       y_pred = decisiontree.predict(X_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous test results
       test_results.append(roc_auc)
    line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()

def evaluate(y_test, y_pred):
    # this block of code returns all the metrics we are interested in
    accuracy = metrics.accuracy_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print ("Accuracy", accuracy)
    print ('F1 score: ', f1)
    print ('ROC_AUC: ' , auc)

def cat_boost():
    print("Start of Cat boost")
    X, y = data_pre_processing()
    X_train, X_test, y_train, y_test = data_processor()
    eval_set = [(X_test, y_test)]
    cb_model = CatBoostClassifier(iterations=1375,
                                 learning_rate= 0.1094999,
                                 depth=6,
                                 thread_count = 10,
                                 eval_metric='AUC',
                                 #eval_metric='Accuracy',
                                 bagging_temperature = 0.9,
                                 od_type='IncToDec',
                                 # l2_leaf_reg= 6,
                                 metric_period = 75,
                                 random_seed = 42,
                                 #logging_level= 'Silent',
                                 random_strength = 1.0,
                                 nan_mode = "Min",
                                 scale_pos_weight = 1.0,
                                 od_wait=100)
    cb_model.fit(X_train, y_train,
                 eval_set=eval_set,
                 cat_features = categorical_features_pos,
                 verbose=True)
    print("Model Evaluation Stage")
    print(cb_model.get_params())
    print("\nevaluate predictions")
    catpred = cb_model.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, catpred)
    f_score = f1_score(y_test, catpred)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))
    print("F1 Score : %.2f%%" % (f_score * 100.0))
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, catpred))
    print('Report : ')
    print(classification_report(y_test, catpred))
    print(mean_squared_error(y_test, catpred))

    # keep probabilities for the positive outcome only
    probs = cb_model.predict_proba(X_test)[:, 1]
    # predict class values
    yhat = cb_model.predict(X_test)
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    # calculate F1 score
    f1 = f1_score(y_test, yhat)
    # calculate precision-recall AUC
    auc_c = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(y_test, probs)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_c, ap))
    plt.figure(figsize=(12, 6))
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', label="No Skill")
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.', label="precision-recall curve")
    # show the plot
    # Line Plot of Precision-Recall Curve
    plt.title("Line Plot of Precision-Recall Curve", {"fontsize": 16});
    plt.ylabel('Precision (y-axis)')
    plt.xlabel('Recall (x-axis)')
    plt.show()

    #Make Prediction and Output for Scoring
    print('Final Result: Make Prediction and Output Score')
    #test_values =pd.read_csv('Data/df_test_enc_2.csv')
    df_test_values_trf = pd.read_csv('../Data/df_test_enc_3.csv')
    # df_test_values_trf = preprocessing.normalize(df_test_values_trf, axis =0)
    # #df_trf = df_trf.astype(int)
    # #df_trf = df_trf.round()
    # #df_enc.dtypes
    # df_test_values_trf = pd.DataFrame(df_test_values_trf,columns = df_trf.columns)


    #df_test_values_trf = clean_dataset(df_test_values_trf)
    # col_names = df_test_values_trf.columns
    # features = df_test_values_trf[col_names]
    # imp = Imputer(strategy="most_frequent").fit(df_test_values_trf)
    # features = imp.transform(df_test_values_trf)

    # scaler = preprocessing.StandardScaler().fit(features)
    # features = scaler.transform(features)
    # df_test_values_trf[col_names] = features
    # cate = df_test_values_trf.columns
    #print(cate)
    #data_norm = preprocessing.normalize(df_test_values_trf, axis = 1)
    #df_test_values_trf = np.concatenate([data_norm])
    #df = pd.DataFrame(df_test_values_trf, columns=cate)

    # this function loops through columns in a data set and defines a predefined scaler to each
    # numeric_columns = ['loan_amount','msa_md', 'state_code', 'lender', 'county_code', 'applicant_income',
    # 'population', 'minority_population_pct','applicant_ethnicity',
    #  'ffiecmedian_family_income', 'tract_to_msa_md_income_pct',
    #  'number_of_owner-occupied_units', 'number_of_1_to_4_family_units']
    # scaler = MinMaxScaler()
    # df_test_values_trf = scale_numeric(df_test_values_trf, numeric_columns, scaler)
    # #df = round(df)
    # # convert all DataFrame columns to the int64 dtype
    # df_test_values_trf = round(df_test_values_trf).astype(int)
    test_values = df_test_values_trf.drop(['Unnamed: 0'],axis=1)
    #test_values = test_values.astype(int)
    test_values=np.array(test_values)
    # Make predictions using the testing set
    cb_pred = cb_model.predict(test_values)
    L_prediccion=pd.DataFrame(data=cb_pred,columns=['accepted'])
    print(L_prediccion.shape)

    L_prediccion.index.names=['row_id']
    L_prediccion['accepted']= L_prediccion['accepted'].astype(np.int64)
    print(L_prediccion.shape)
    print(L_prediccion.head())
    L_prediccion.to_csv('../Data/submission_1.csv')
    print("End of Cat boost")

def xgb_classifier(X_train, X_test, y_train, y_test, useTrainCV=False, cv_folds=5, early_stopping_rounds=20):
    """
    # {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3} 0.862920874517388
    # {'colsample_bytree': 1.0, 'gamma': 0.2} 0.871
    # {'gamma': 0.2, 'scale_pos_weight': 1} 0.8702009952422571
    # {'subsample': 0.6} 0.864310306628855
    """
    print("Start of XGB classifier")
    alg = xgb.XGBClassifier(learning_rate=0.1195, n_estimators=1200, max_depth=75,
                        min_child_weight=6, gamma=0.2, subsample=0.6, colsample_bytree=1.0,num_leaves= 400,
                        objective='binary:logistic', random_state=42, nthread=4, scale_pos_weight=1, seed=27)

    print('Start Training')
    eval_set = [(X_test, y_test)]
    alg.fit(X_train, y_train, eval_metric='auc', eval_set=eval_set, verbose=1)

    #param_test1 = {'max_depth':[4,5,6,8], 'min_child_weight':[4,5,6]}

    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth=5,
    #                                                 min_child_weight=3, gamma=0.2, subsample=0.8,
    #                                                 colsample_bytree=1.0,
    #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                 seed=27),
    #                         param_grid=param_test1,
    #                         scoring='f1',
    #                         n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(X_train, y_train)
    # print("search grid1 : ",gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
    # print("search grid4 : ",gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    print("Start Predicting")
    predictions = alg.predict(X_test)
    pred_proba = alg.predict_proba(X_test)[:, 1]
    predictions = [round(value) for value in predictions]
    print("\nevaluate predictions")
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, pred_proba)
    f_score = f1_score(y_test, predictions)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))
    print("AUC Score : %.2f%%" % (roc * 100.0))
    print("F1 Score : %.2f%%" % (f_score * 100.0))
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, predictions))
    print('Report : ')
    print(classification_report(y_test, predictions))
    print(mean_squared_error(y_test, predictions))


    #Make Prediction and Output for Scoring
    print('Final Result: Make Prediction and Output Score')
    #test_values =pd.read_csv('Data/df_test_enc_2.csv')
    df_test_values_trf = pd.read_csv('../Data/df_test_enc_3.csv')
    # df_test_values_trf = preprocessing.normalize(df_test_values_trf, axis =0)
    # #df_trf = df_trf.astype(int)
    # #df_trf = df_trf.round()
    # #df_enc.dtypes
    # df_test_values_trf = pd.DataFrame(df_test_values_trf,columns = df_trf.columns)


    #df_test_values_trf = clean_dataset(df_test_values_trf)
    # col_names = df_test_values_trf.columns
    #features = df_test_values_trf[col_names]
    # imp = Imputer(strategy="most_frequent").fit(df_test_values_trf)
    # features = imp.transform(df_test_values_trf)
    #
    # scaler = preprocessing.StandardScaler().fit(features)
    # features = scaler.transform(features)
    # df_test_values_trf[col_names] = features
    test_values = df_test_values_trf.drop(['Unnamed: 0'],axis=1)
    #test_values = test_values.astype(np.float64)
    test_values=np.array(test_values)
    # Make predictions using the testing set
    rf_pred = alg.predict(test_values)
    L_prediccion=pd.DataFrame(data=rf_pred,columns=['accepted'])
    print(L_prediccion.shape)

    L_prediccion.index.names=['row_id']
    L_prediccion['accepted']= L_prediccion['accepted'].astype(np.int64)
    print(L_prediccion.shape)
    print(L_prediccion.head())
    L_prediccion.to_csv('../Data/submission.csv')
    print("End of XGB classifier")

def cross_validation():
    X, y = data_pre_processing()
    print("Start of Xgboost cross validation")
    # CV model
    model_ord = xgb.XGBClassifier()
    kfold = KFold(n_splits=5, random_state=54321)
    results = cross_val_score(model_ord, X, y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    print("End of Xgboost cross validation")

    print("Start of Xgboost StratifiedKFold cross validation")
    # CV model
    model_stra = xgb.XGBClassifier()
    kfold = StratifiedKFold(n_splits=5, random_state=54321)
    results = cross_val_score(model_stra, X, y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    print("End of Xgboost StratifiedKFold cross validation")

def logistic_regression():
    """
    F1 score is: 0.7285714285714285
    AUC Score is: 0.9667565771367231
    """
    print("Start of logistic")
    X_train, X_test, y_train, y_test = data_processor()
    clf = LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print("Logistic Regression Score: ", clf.score(X_test, y_test))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))
    print("Logistic Regression F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("Logistic Regression AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
    print("End of logistic regression")


def logistic_with_smote():
    print("Start of logist with smote")
    X_train, X_test, y_train, y_test = data_processor()
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    # build model with SMOTE imblearn
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=42), clf)
    smote_model = smote_pipeline.fit(X_train, y_train)
    smote_prediction = smote_model.predict(X_test)
    smote_prediction_proba = smote_model.predict_proba(X_test)[:, 1]
    print(classification_report_imbalanced(y_test, smote_prediction))
    print('SMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))
    print("SMOTE AUC score: ", roc_auc_score(y_test, smote_prediction_proba))
    print("SMOTE F1 Score: ", f1_score(y_test, smote_prediction))
    print("End of logistic smote")


def randomForest():
    """
    F1 score is: 0.7857142857142857
    AUC Score is: 0.9450972761670293
    """
    print("Start of Random forest")
    X_train, X_test, y_train, y_test = data_processor()
    # parameters = {'n_estimators': [10, 20, 30, 50], 'max_depth': [2, 3, 4]}

    clf = RandomForestClassifier(max_depth=6, n_estimators=800)
    # clf = GridSearchCV(alg, parameters, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Random Forest Score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    print("\nevaluate predictions")
    print("Random Forest F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("Random Forest AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    f_score = f1_score(y_test, y_pred)
    print("Random Forest Accuracy : %.2f%%" % (accuracy * 100.0))
    print("Random Forest AUC Score is : %.2f%%" % (roc * 100.0))
    print("Random Forest F1 Score : %.2f%%" % (f_score * 100.0))
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, y_pred))
    print('Report : ')
    print(classification_report(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))


def neural_nets():
    """
    Score:  0.9994148145547324
    F1 score is: 0.822695035460993
    AUC Score is: 0.9608730286337007
    """
    X_train, X_test, y_train, y_test = data_processor()
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100,))

    clf.fit(X_train, y_train)
    print("Neural Network Score: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print("Neural Network F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("Neural Network AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))

def xgb_regressor():
    print("Start of xgb regressor")
    X, y = data_pre_processing()
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X):
        xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
        predictions = xgb_model.predict(X[test_index])
        actuals = y[test_index]
        predictions = [round(value) for value in predictions]
        # evaluate predictions
        # evaluate predictions
        accuracy = accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print('Confusion Matrix :')
        print(confusion_matrix(actuals, predictions))
        print('Report : ')
        print(classification_report(actuals, predictions))
        print(mean_squared_error(actuals, predictions))
    print("Parameter optimization")
    xgb_model = xgb.XGBRegressor()
    crf = GridSearchCV(xgb_model,
                       {'max_depth': [6,8],'n_estimators': [500,855],'learning_rate': [0.1,0.1095]}, verbose=1)
    crf.fit(X,y)
    print(crf.best_score_)
    print(crf.best_params_)

    print("End of regressor")

    # The sklearn API models are picklable
    print("Pickling sklearn API models")
    # must open in binary format to pickle
    pickle.dump(crf, open("Data/best_predict.pkl", "wb"))
    clf2 = pickle.load(open("Data/best_predict.pkl", "rb"))
    print(np.allclose(crf.predict(X), clf2.predict(X)))

def random_forest_classifier():
    print("Random forest binary classification")
    X, y = data_pre_processing()
    # Build random forest classifier (same config)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X):
        rf_clf = RandomForestClassifier().fit(X[train_index], y[train_index])
        # make predictions for test data
        predictions = rf_clf.predict(X[test_index])
        actuals = y[test_index]
        predictions = [round(value) for value in predictions]
        # evaluate predictions
        accuracy = accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print('Confusion Matrix :')
        print(confusion_matrix(actuals, predictions))
        print('Report : ')
        print(classification_report(actuals, predictions))


    nr.seed(3456)
    ## Define the dictionary for the grid search and the model object to search on
    param_grid_Forest = {"n_estimators": [500,1000],
                         'max_features':[6,8,12],
                        'max_depth': [2,4,6]}

    ## Perform the grid search over the parameters

    Grid_Forest = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid_Forest, verbose=1)


    #'max_depth': [2,4,6],'n_estimators': [5,3,8],'learning_rate': [0.1,0.01],
    Grid_Forest.fit(X,y)
    print(Grid_Forest.best_score_)
    print(Grid_Forest.best_params_)

def auc2(m, train, test):
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))


def LGBMClassifier():
    X_train, X_test, y_train, y_test = data_processor()
    lg = lgb.LGBMClassifier(learning_rate=0.01195, n_estimators=1400, max_depth=75,
                        min_child_weight=6,subsample=0.4, colsample_bytree=1.0,
                        objective='binary', random_state=54321, num_leaves= 400, nthread=4, scale_pos_weight=1, seed=27, silent=False)


    # param_dist = {"max_depth": [25,50, 75],
    #               "learning_rate" : [0.01,0.05,0.1],
    #               "num_leaves": [300,900,1200],
    #               "n_estimators": [200]
    #              }
    # grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
    # grid_search.fit(X_train,y_train)
    # grid_search.best_estimator_
    #
    # d_train = lgb.Dataset(X_train, label=y_train)
    # params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300}
    #
    #
    # # Without Categorical Features
    # model1 = lgb.train(params, d_train)
    # #auc2(model2, train, test_values)
    # #With Catgeorical Features
    # model2 = lgb.train(params, d_train, categorical_feature = categorical_features_pos)
    # #auc2(model2, train, test_values)
    # print('Final Result: Make Prediction and Output Score')
    # df_test_values_trf = pd.read_csv('../Data/df_test_enc_3.csv')
    # test_values = df_test_values_trf.drop(['Unnamed: 0'],axis=1)
    # test_values=np.array(test_values)
    # # Without Categorical Features
    # model2 = lgb.train(params, d_train)
    # auc2(model2, train, test_values)
    # #With Catgeorical Features
    # model2 = lgb.train(params, d_train, categorical_feature = categorical_features_pos)
    # auc2(model2, train, test_values)

    eval_set = [(X_test, y_test)]
    lg.fit(X_train, y_train,
                 eval_set=eval_set, eval_metric='auc', verbose=True)

    print(lg.get_params())
    print("\nevaluate predictions")
    print("LGBMClassifier Score: ", lg.score(X_test, y_test))
    y_pred = lg.predict(X_test)
    y_pred_proba = lg.predict_proba(X_test)[:, 1]

    print("\nLGBMClassifier evaluate predictions")
    print("LGBMClassifier F1 score is: {}".format(f1_score(y_test, y_pred)))
    print("LGBMClassifier ROC AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    f_score = f1_score(y_test, y_pred)
    print("LGBMClassifier Accuracy : %.2f%%" % (accuracy * 100.0))
    print("LGBMClassifier ROC Score is : %.2f%%" % (roc * 100.0))
    print("LGBMClassifier F1 Score : %.2f%%" % (f_score * 100.0))
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, y_pred))
    print('Report : ')
    print(classification_report(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))

    # predict probabilities
    probs =lg.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = lg.predict_proba(X_test)[:, 1]
    # calculate AUC
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # show the plot
    pyplot.show()

    # #Make Prediction and Output for Scoring
    # print('Final Result: Make Prediction and Output Score')
    # #test_values =pd.read_csv('Data/df_test_enc_2.csv')
    # df_test_values_trf = pd.read_csv('../Data/df_test_enc_3.csv')
    # test_values = df_test_values_trf.drop(['Unnamed: 0'],axis=1)
    # #test_values = test_values.astype(np.float64)
    # test_values=np.array(test_values)
    # # Make predictions using the testing set
    # lg_pred = lg.predict(test_values)
    # L_prediccion=pd.DataFrame(data=lg_pred,columns=['accepted'])
    # print(L_prediccion.shape)
    #
    # L_prediccion.index.names=['row_id']
    # L_prediccion['accepted']= L_prediccion['accepted'].astype(np.int64)
    # print(L_prediccion.shape)
    # print(L_prediccion.head())
    # L_prediccion.to_csv('../Data/submission_4.csv')
    # print("End of LGB classifier")


if __name__ == "__main__":
    start = time.time()
    X_train, X_test, y_train, y_test = data_processor()
    print("baseline")
    #cross_validation()
    getbaseline()
    # xgb_classifier(X_train, X_test, y_train, y_test)
    # LGBMClassifier()
    # logistic_regression()
    # randomForest()
    # logistic_with_smote()
    # neural_nets()
    cat_boost()
    #random_forest_classifier()
    # xgb_regressor()
    #plot_baseline()
    #best_festure()
    #df = data_transform()
    #plot_best_feature(df,X_train,y_train)
    print("Total Time is: ", (time.time() - start)/60)
