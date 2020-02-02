#
# # coding: utf-8
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import RobustScaler,Normalizer, MinMaxScaler,FunctionTransformer, PolynomialFeatures, Imputer
import category_encoders as ce
# get_ipython().magic('matplotlib inline')
plt.ion
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
import warnings
warnings.filterwarnings('ignore')

# # Input data files are available in the "../Data/" directory.
# ## Prepare Dataset for Models
# Drop Row ID from training set as it is not a predictor.
# Load test dataset
DATA_File = os.listdir('../Data')[0]
print(os.listdir('../Data'))
df_test_dataset = pd.read_csv('../Data/test_values.csv')
print("Home mortgage test data values has {} data points with {} variables each.".format(*df_test_dataset.shape))
print(df_test_dataset.shape)
print(df_test_dataset.head())
print(df_test_dataset.isnull().sum())


print(df_test_dataset['co_applicant'].unique())
def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()

categorical_columns= ['co_applicant']

for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_co_applicant = np.concatenate([temp], axis = 1)
#explorer categorical (bool) columns
enc_co_applicant = ['co_applicant_True', 'co_applicant_False']
print(Features_co_applicant.shape)
print(Features_co_applicant[:10, :])

print(df_test_dataset['loan_type'].unique())
categorical_columns= ['loan_type']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_loan_type = np.concatenate([temp], axis = 1)
enc_loan_type = ['loan_type_conv','loan_type_FHA','loan_type_VA','loan_type_FSA_RHS']
print(Features_loan_type.shape)
print(Features_loan_type[:5, :])

print(df_test_dataset['property_type'].unique())
categorical_columns= ['property_type']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_property_type = np.concatenate([temp], axis = 1)
enc_property_type = ['property_type_One_to_four_family','property_type_Manufactured_housing', 'property_type_Multifamily']
print(Features_property_type.shape)
print(Features_property_type[:4, :])

print(df_test_dataset['loan_purpose'].unique())
categorical_columns= ['loan_purpose']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_loan_purpose = np.concatenate([temp], axis = 1)
enc_loan_purpose = ['loan_purpose_Home_purchase','loan_purpose_Home_improvement','loan_purpose_Refinancing']
print(Features_loan_purpose.shape)
print(Features_loan_purpose[:4, :])

print(df_test_dataset['occupancy'].unique())
categorical_columns= ['occupancy']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_occupancy = np.concatenate([temp], axis = 1)
enc_occupancy = ['occupancy_Owner_occupied','occupancy_Not_owner_occupied','occupancy_Not_applicable']
print(Features_occupancy.shape)
print(Features_occupancy[:4, :])

print(df_test_dataset['preapproval'].unique())
categorical_columns= ['preapproval']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_preapproval = np.concatenate([temp], axis = 1)
enc_preapproval = ['preapproval_Preapproval_requested','preapproval_Preapproval_not_requested','preapproval_Not_applicable']
print(Features_preapproval.shape)
print(Features_preapproval[:4, :])

print(df_test_dataset['applicant_ethnicity'].unique())
categorical_columns= ['applicant_ethnicity']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_applicant_ethnicity = np.concatenate([temp], axis = 1)
enc_applicant_ethnicity = ['applicant_ethnicity_Hispanic_Latino','applicant_ethnicity_Not_Hispanic_Latino',
                           'applicant_ethnicity_Information_not_provided','applicant_ethnicity_Not_applicable']
print(Features_applicant_ethnicity.shape)
print(Features_applicant_ethnicity[:5, :])

print(df_test_dataset['applicant_race'].unique())
categorical_columns= ['applicant_race']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_applicant_race = np.concatenate([temp], axis = 1)
enc_applicant_race = ['applicant_race_American_Indian','applicant_race_Asian', 'applicant_race_African_American',
           'applicant_race_Native_Hawaiian','applicant_race_White','applicant_race_Information_not_provided',
           'applicant_race_Not_applicable']
print(Features_applicant_race.shape)
print(Features_applicant_race[:7, :])

print(df_test_dataset['applicant_sex'].unique())
categorical_columns= ['applicant_sex']
for col in categorical_columns:
    temp = encode_string(df_test_dataset[col])
    Features_applicant_sex = np.concatenate([temp], axis = 1)
enc_applicant_sex = ['applicant_sex_Male','applicant_sex_Female', 'applicant_sex_Information_not_provided',
                 'applicant_sex_Not_applicable']
print(Features_applicant_sex.shape)
print(Features_applicant_sex[:5, :])

# df_test_dataset['loan_amount_log'] = np.log((1 + df_test_dataset['loan_amount']))
# df_test_dataset['applicant_income_log'] = np.log((1 + df_test_dataset['applicant_income']))
# df_test_dataset['ffiecmedian_family_income_log'] = np.log((1 + df_test_dataset['ffiecmedian_family_income']))
# df_test_dataset['tract_to_msa_md_income_pct_log'] = np.log((1 + df_test_dataset['tract_to_msa_md_income_pct']))
# df_test_dataset['number_of_owner-occupied_units_log'] = np.log((1 + df_test_dataset['number_of_owner-occupied_units']))
# df_test_dataset['number_of_1_to_4_family_units_log'] = np.log((1 + df_test_dataset['number_of_1_to_4_family_units']))
# #df_test_dataset['lender_log'] = np.log((1 + df_test_dataset['lender']))
# #df_test_dataset['msa_md_log'] = np.log((1 + df_test_dataset['msa_md']))
# #df_test_dataset['state_code_log'] = np.log((1 + df_test_dataset['state_code']))
# #df_test_dataset['county_code_log'] = np.log((1 + df_test_dataset['county_code']))
# df_test_dataset['minority_population_pct_log'] = np.log((1 + df_test_dataset['minority_population_pct']))
# df_test_dataset['population_log'] = np.log((1 + df_test_dataset['population']))

# To see and get number columns
category_num_cols_logs =(df_test_dataset.dtypes == float) | (df_test_dataset.dtypes==np.int64)
raw_category_num_cols_logs = [c for c in category_num_cols_logs.index if category_num_cols_logs[c]]
raw_category_num_cols_logs

cols = raw_category_num_cols_logs + enc_co_applicant + enc_loan_type + enc_property_type + enc_loan_purpose + enc_occupancy + enc_preapproval + enc_applicant_ethnicity + enc_applicant_race + enc_applicant_sex
#cols = raw_category_num_cols_logs
len(cols)
cols

df_test_enc= np.concatenate([df_test_dataset[raw_category_num_cols_logs],Features_co_applicant,Features_loan_type,Features_property_type,Features_loan_purpose,Features_occupancy,Features_preapproval,Features_applicant_ethnicity,Features_applicant_race,Features_applicant_sex],axis=1)
df_test_enc = pd.DataFrame(df_test_enc, columns=cols)


#select the cols of interest and assign back to the df:
print("If isfinite")
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

print(np.all(np.isfinite(df_test_enc)))
#print(np.any(np.isfinite(df_enc)))

def replace_space(df):
    return df.replace(r'^\s*$', np.nan, regex=True)

df_test_enc = clean_dataset(df_test_enc)



# #replace: np.inf, -np.inf

print("Check if isfinite")
df_test_enc['msa_md'] = df_test_enc['msa_md'].replace([np.inf, -np.inf], [df_test_enc['msa_md'].mean(), df_test_enc['msa_md'].mean()])
df_test_enc['state_code'] = df_test_enc['state_code'].replace([np.inf, -np.inf], [df_test_enc['state_code'].mean(), df_test_enc['state_code'].mean()])
df_test_enc['county_code'] = df_test_enc['county_code'].replace([np.inf, -np.inf],  [df_test_enc['county_code'].mean(), df_test_enc['county_code'].mean()])
df_test_enc['msa_md'] = df_test_enc['msa_md'].replace(-1, df_test_enc['msa_md'].mode())
df_test_enc['state_code'] = df_test_enc['state_code'].replace(-1, df_test_enc['state_code'].mode())
df_test_enc['county_code'] = df_test_enc['county_code'].replace(-1, df_test_enc['county_code'].mode())

# #Imputing Missing values with mean for continuous variable
df_test_enc['loan_amount'].fillna(df_test_enc['loan_amount'].median(), inplace=True)
df_test_enc['applicant_income'].fillna(df_test_enc['applicant_income'].median(), inplace=True)
df_test_enc['population'].fillna(df_test_enc['population'].median(), inplace=True)
df_test_enc['minority_population_pct'].fillna(df_test_enc['minority_population_pct'].median(), inplace=True)
df_test_enc['ffiecmedian_family_income'].fillna(df_test_enc['ffiecmedian_family_income'].median(), inplace=True)
df_test_enc['tract_to_msa_md_income_pct'].fillna(df_test_enc['tract_to_msa_md_income_pct'].median(), inplace=True)
df_test_enc['number_of_owner-occupied_units'].fillna(df_test_enc['number_of_owner-occupied_units'].median(), inplace=True)
df_test_enc['number_of_1_to_4_family_units'].fillna(df_test_enc['number_of_1_to_4_family_units'].median(), inplace=True)
df_test_enc['msa_md'].fillna(df_test_enc['msa_md'].median(), inplace=True)
df_test_enc['state_code'].fillna(df_test_enc['state_code'].median(), inplace=True)
df_test_enc['county_code'].fillna(df_test_enc['county_code'].median(), inplace=True)
df_test_enc['lender'].fillna(df_test_enc['lender'].median(), inplace=True)


#
#
# df_test_enc['loan_amount_ex'] = df_test_enc['loan_amount']
# df_test_enc['applicant_income_ex'] = df_test_enc['applicant_income']
# df_test_enc['population_ex'] = df_test_enc['population']
# df_test_enc['minority_population_pct_ex'] = df_test_enc['minority_population_pct']
# df_test_enc['ffiecmedian_family_income_ex'] = df_test_enc['ffiecmedian_family_income']
# df_test_enc['tract_to_msa_md_income_pct_ex'] = df_test_enc['tract_to_msa_md_income_pct']
# df_test_enc['number_of_owner-occupied_units_ex'] = df_test_enc['number_of_owner-occupied_units']
# df_test_enc['number_of_1_to_4_family_units_ex'] = df_test_enc['number_of_1_to_4_family_units']
# df_test_enc['msa_md_ex'] = df_test_enc['msa_md']
# df_test_enc['state_code_ex'] = df_test_enc['state_code']
# df_test_enc['county_code_ex'] = df_test_enc['county_code']
# df_test_enc['lender_ex'] = df_test_enc['lender']
#
#
# #
# def scale_numeric(data, numeric_columns, scaler):
#     for col in numeric_columns:
#         data[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
#     return data
# #
# # numeric_columns = ['msa_md_ex', 'state_code_ex', 'county_code_ex', 'loan_amount_ex', 'applicant_income_ex',
# #                     'lender_ex', 'population_ex', 'minority_population_pct_ex',
# #                      'ffiecmedian_family_income_ex', 'tract_to_msa_md_income_pct_ex',
# #                      'number_of_owner-occupied_units_ex', 'number_of_1_to_4_family_units_ex']
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # df_test_enc = scale_numeric(df_test_enc, numeric_columns, scaler)
# #
#
# df_test_encoding = df_test_enc
# print("\nEncode Variables")
# categorical = ['msa_md', 'state_code', 'county_code', 'lender']
# print("Encoding :",categorical)
#
# enc = preprocessing.LabelEncoder()
# enc.fit(categorical)
# enc_cat_feature = enc.transform(categorical)
# # Encoder:# instantiate an encoder - here we use Binary()
# ce_binary = ce.BinaryEncoder(enc_cat_feature)
# # fit and transform and presto, you've got encoded data
# df_test_enc = ce_binary.fit_transform(df_test_encoding)

# # # df_test_enc = df_test_enc.drop(['row_id','loan_type', 'property_type', 'loan_purpose', 'occupancy', 'preapproval',
# # # 'applicant_ethnicity', 'applicant_race', 'applicant_sex'], axis=1)
#
# cols = df_test_enc.columns.tolist()
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_test_enc = scale_numeric(df_test_enc, cols, scaler)


df_test_enc = df_test_enc.drop(['row_id'],axis=1)
print(np.all(np.isfinite(df_test_enc)))
df_test_enc = round(df_test_enc).astype(int)
print(df_test_enc.dtypes)
print(df_test_enc.head())
#cols = df_test_enc.columns.tolist()
print(cols)
print(df_test_enc.shape)
df_test_enc.to_csv('../Data/df_test_enc_3.csv')
