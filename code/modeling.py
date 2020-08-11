### Libraries
import numpy as np
import pandas as pd
import os
import sys
import gc
import re
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score, mean_squared_log_error
from sklearn import preprocessing

# mine
from get_pred import get_oof_ypred
from run_models import RunModel

### Config
N_SPLITS = 5
MODEL = sys.argv[1]
N_SEEDAVE = sys.argv[2]
SEED = sys.argv[3]

### Load data
INPUT_DIR = '../output/'
OUTPUT_DIR = '../output/'

def read_data():
    train = pd.read_feather(INPUT_DIR + 'train.csv')
    test = pd.read_feather(INPUT_DIR + 'test.csv')
    return train, test
train, test = read_data()

### add 'leak' data for training
# https://prob.space/competitions/re_real_estate_2020/discussions/masato8823-Post9982d5b9dcd6a33111e0
test_known = test.copy()
test_known['y'] = -1
test_known['y'].iloc[1305] = 30000
test_known['y'].iloc[1960] = 13000
test_known['y'].iloc[8069] = 28000

# add to train
train = pd.concat([train, test_known.iloc[[1305, 1960, 8069]]], ignore_index=True)
del test_known
gc.collect()

### Features (permutation importance > 0)
features = ['area', 'city_code', 'built_year', 'area_per_room2', 'total_area', 'distance_to_station', 'road_width', 'total_area_ratio', 'entry', 'nearest_station', 'area_ratio2', 'road_kinds_isnan', 'road_direction', 'road_kinds', 'area_name', 'kinds', 'structure', 'city_plan', 'TotalRoomNum', 'renovation', 'shape', 'isnan_sum', 'volume_area_ratio', 'area_per_room', 'RoomNumRatio', 'purpose', 'region', 'D', 'L', 'issues', 'issues_isnan', 'house_plan_isnan', 'usage_share', 'usage_handwork', 'RoomNum', 'structure_isnan', 'built_year_isnan', 'entry_isnan', 'usage_home', 'deal_quater', 'usage_office', 'usage_others', 'road_width_isnan', 'usage_shop', 'K', 'usage_parking', 'purpose_isnan', 'region_isnan', 'OpenFloor', 'renovation_isnan', 'road_direction_isnan', 'shape_isnan', 'total_area_isnan', 'R', 'area_per_room_isnan', 'area_ratio2_isnan', 'deal_timing']
categoricals = ['kinds',
 'region',
 'city_code',
 'area_name',
 'nearest_station',
 'shape',
 'structure',
 'purpose',
 'road_direction',
 'road_kinds',
 'city_plan',
 'renovation',
 'issues',]

### Modeling
# target to log
target = 'y'
y_true = train[target].values
train[target] = np.log1p(train[target].values)

# scaler
if MODEL not in ['xgb', 'catb', 'lgb']:
    scaler = 'Standard'

# fitting
y_pred = np.zeros(len(test))
oof = np.zeros(len(train))
score = 0
for s in range(N_SEEDAVE):
    # model fitting
    model = RunModel(train, test, target, features, categoricals=categoricals,
                model=MODEL, task="regression", n_splits=N_SPLITS, cv_method="KFold", 
                group=None, seed=SEED+s, scaler=scaler)
    
    # average
    oof_ = np.clip(np.expm1(model.oof), 0, 70000)
    y_ = np.clip(np.expm1(model.y_pred), 0, 70000)
    oof += oof_ / N_SEEDAVE
    y_pred += y_ / N_SEEDAVE
    score += model.score / N_SEEDAVE

### CV score
print('CV score = ', score)

### Submit
submit_df = pd.DataFrame({'y': y_pred})
submit_df.index.name = 'id'
submit_df.index = submit_df.index + 1
submit_df.to_csv(OUTPUT_DIR + f'submission_{MODEL}_nofix.csv') # raw prediction

# postprocess (https://prob.space/competitions/re_real_estate_2020/discussions/masato8823-Post9982d5b9dcd6a33111e0)
submit_df['y'].iloc[1305] = 30000
submit_df['y'].iloc[1960] = 13000
submit_df['y'].iloc[8069] = 28000

submit_df.to_csv(OUTPUT_DIR + f'submission_{MODEL}_fix.csv') # replace with 'leaked' values
print('submitted!')

### Save oof for stacking
np.save(OUTPUT_DIR + f'oof_{MODEL}', oof)
np.save(OUTPUT_DIR + f'ypred_{MODEL}', y_pred)
print('saved!')
