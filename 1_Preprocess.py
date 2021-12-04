################################################################
#### 1_Preprocess                                           ####
################################################################

import os
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import operator
from functools import reduce

from google.colab import drive

# Load data
train_data_path = "./train_preliminary/"
#train_final_data_path = "./train_semi_final/"
test_data_path = "./test/"
save_path = './processed_data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create the ad, click log and user tables
df_ad = pd.concat([pd.read_csv(os.path.join(train_data_path, "ad.csv")), pd.read_csv(os.path.join(test_data_path, "ad.csv"))])
df_click = pd.concat([pd.read_csv(os.path.join(train_data_path, "click_log.csv")), pd.read_csv(os.path.join(test_data_path, "click_log.csv"))])    
df_train_user = pd.read_csv(os.path.join(train_data_path, "user.csv"))

################################################################
#### Process train_user data                                ####
################################################################
# Get index where user_id < 120000
select_index = np.array(np.where(df_train_user['user_id']<=120000))
select_index = reduce(operator.add,select_index)
# update the user table with the selected user_id
df_train_user_new = df_train_user.iloc[select_index,]
del df_train_user

################################################################
#### Process df_ad data                                     ####
################################################################
# remove duplicates & deal with NaN
df_ad = df_ad.drop_duplicates()
df_ad[df_ad=="\\N"] = np.nan
df_ad.fillna(0, inplace=True)
df_ad = df_ad.astype(int)

# maxfill missing value and 0-pad
df_ad.loc[df_ad['product_id']==0, 'product_id'] = df_ad['product_id'].max() + 1
df_ad.loc[df_ad['industry']==0, 'industry'] = df_ad['industry'].max() + 1

################################################################
#### Merge ad data with click log                           ####
################################################################
# df_click_new shows the ad clicks grouped by user
df_click = df_click.merge(df_ad, on='creative_id')
# sort and make sure the entries are ordered by time
df_click.sort_values(by=['user_id', 'time'], inplace=True)
df_click = df_click.astype(np.int32)

################################################################
#### Process df_click data                                  ####
################################################################
# Get index where user_id index < 120000
select_index=np.array(np.where(df_click['user_id']<=120000))
select_index = reduce(operator.add,select_index)
df_click_new=df_click.iloc[select_index,:]
print(df_click_new.max())
print(df_click_new.nunique())

# Get index where creative_id index < 1000000
select_index=np.array(np.where(df_click_new['creative_id']<=1000000))
select_index = reduce(operator.add,select_index)
df_click_new=df_click_new.iloc[select_index,:]
print(df_click_new.max())
print('\n')
print(df_click_new.nunique())

# Get index where ad_id index < 1000000
select_index=np.array(np.where(df_click_new['ad_id']<=1000000))
select_index = reduce(operator.add,select_index)
df_click_new=df_click_new.iloc[select_index,:]
print(df_click_new.max())
print('\n')
print(df_click_new.nunique())

# Get index where product_id index < 1000000
select_index=np.array(np.where(df_click_new['product_id']<=20000))
select_index = reduce(operator.add,select_index)
df_click_new=df_click_new.iloc[select_index,:]
print(df_click_new.max())
print('\n')
print(df_click_new.nunique())

## Uncomment to view the vocabulary size of each feature
# df_click_new.max()
# df_click_new.nunique()

## Group by user_id, and put the click log of each user together into a pandas series
def process_group(df):
    dic = {}
    for name in ['time', 'creative_id', 'click_times', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        dic[name] = df[name].values
    return pd.Series(dic)

df_click_group = df_click_new.groupby('user_id').progress_apply(process_group)

## Merge age and gender into df_train
df_click_group = df_click_group.join(df_train_user_new.set_index('user_id'))

## save as a pickle file
df_click_group.to_pickle(os.path.join(save_path, 'processed_data_numerical_new.pkl'))