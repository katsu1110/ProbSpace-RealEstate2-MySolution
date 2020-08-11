### Libraries
import numpy as np
import pandas as pd
import os
import sys
import gc
import re
from sklearn import preprocessing

### Load data
input_path = "../input/"
output_path = "../output/"

def read_data():
    train = pd.read_csv(input_path + 'train_data.csv')
    test = pd.read_csv(input_path + 'test_data.csv')
    return train, test
train, test = read_data()

### drop columns
drops = ['都道府県名', '市区町村名',]
train.drop(columns=drops, inplace=True)
test.drop(columns=drops, inplace=True)

### Feature engineering
#   Largely adapted from https://prob.space/competitions/re_real_estate_2020/discussions/Oregin-Post8b7bae773e3db3bbba97
#   by Oregin

def madori_converter(df):
    """
    Madori (house-plan) converter
    """
    df['L'] = df['間取り'].map(lambda x: 1 if 'Ｌ' in str(x) else 0)
    df['D'] = df['間取り'].map(lambda x: 1 if 'Ｄ' in str(x) else 0)
    df['K'] = df['間取り'].map(lambda x: 1 if 'Ｋ' in str(x) else 0)
    df['S'] = df['間取り'].map(lambda x: 1 if 'Ｓ' in str(x) else 0)
    df['R'] = df['間取り'].map(lambda x: 1 if 'Ｒ' in str(x) else 0)
    df['Maisonette'] = df['間取り'].map(lambda x: 1 if 'メゾネット' in str(x) else 0)
    df['OpenFloor'] = df['間取り'].map(lambda x: 1 if 'オープンフロア' in str(x) else 0)
    df['Studio'] = df['間取り'].map(lambda x: 1 if 'スタジオ' in str(x) else 0)
    df['Special'] = df['Maisonette'] + df['OpenFloor'] + df['Studio'] 
    df['RoomNum'] = df['間取り'].map(lambda x: re.sub("\\D", "", str(x)))
    df['RoomNum'] = df['RoomNum'].map(lambda x:int(x) if x!='' else 0)
    df['TotalRoomNum'] = df[['L', 'D', 'K', 'S', 'R', 'RoomNum']].sum(axis=1)
    df['RoomNumRatio'] = df['RoomNum'] / df['TotalRoomNum']                           
    return df

def change_to_number(df,input_column_name,output_column_name):
    df[output_column_name] = df[input_column_name].map(lambda x: re.sub(r'([0-9]+)m\^2未満', '9', str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x: re.sub("\\D", "", str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x:int(x) if x!='' else np.nan)
    return df

def change_to_minute(df,input_column_name,output_column_name):
    df[output_column_name] = df[input_column_name].map(lambda x: re.sub(r"30分\?60分", "45", str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x: re.sub(r"2H\?", "120", str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x: re.sub(r"1H30\?2H", "105", str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x: re.sub(r"1H\?1H30", "75", str(x))) 
    df[output_column_name] = df[output_column_name].map(lambda x: re.sub("\\D", "", str(x)))
    df[output_column_name] = df[output_column_name].map(lambda x:int(x) if x!='' else np.nan)
    return df

def change_to_float(df,input_column_name,output_column_name):
    # ５０ｍ以上は５１にする
    df[output_column_name] = df[input_column_name].map(lambda x: re.sub("50.0m以上", "51.0", str(x)))
    #数値にする（Nullの場合はー１にする）
    df[output_column_name] = df[output_column_name].map(lambda x:float(x) if x!='nan' else np.nan)
    
    return df

### My feature engineering

# nan as a feature (missing record itself could be meaninigful?)
def onehot_nan(df, features):
    """
    add 'isnan_' features
    """
    nrows = len(df)
    isnan_features = []
    for f in features:
        if df[f].isna().sum() > nrows / 5:
            df[f + '_isnan'] = 0
            df.loc[df[f].isna() == True, f + '_isnan'] = 1
            isnan_features.append(f + '_isnan')
    df['isnan_sum'] = df[isnan_features].sum(axis=1)
    isnan_features.append('isnan_sum')
    return df, isnan_features

# label encoding
def label_encoding(x_train, x_test, cat_features):
    """
    label encoding object features
    """
    for c in cat_features:
        # only use categorical value shared between train and test
        print(c)
        shared = list(set(x_train[c].unique().tolist()) & set(x_test[c].unique().tolist()))
        x_train.loc[~x_train[c].isin(shared), c] = np.nan
        x_test.loc[~x_test[c].isin(shared), c] = np.nan

        # label encoding
        le = preprocessing.LabelEncoder()           
        x_train[c] = x_train[c].fillna("NaN")
        x_test[c] = x_test[c].fillna("NaN")
        x_train[c] = le.fit_transform(x_train[c].astype(str))
        x_test[c] = le.transform(x_test[c].astype(str))
        x_train[c] = x_train[c].astype(int)
        x_test[c] = x_test[c].astype(int)
    return x_train, x_test, cat_features

def kenchiku_convert(x):
    try:
        if x[:2] == '昭和':
            return 1926 + int(x[2:-1])
        elif x[:2] == '平成':
            return 1989 + int(x[2:-1])
        elif x == '戦前':
            return 1925
    except:
        return np.nan
        
def jiten_convert(x):
    """
    Bin 'deal timing'
    """
    try:
        v = int(x[:4]) + float(x[6]) / 4
    except:
        v = np.nan    
    return v

def torihiki_quater(x):
    """
    Just extract quarter of 'deal timing'
    """
    try:
        return int(x[6])
    except:
        return np.nan
    
def usage_feats(df):
    df['用途'] = df['用途'].fillna('その他')
    df['usage_others'] = df['用途'].apply(lambda x : 1 if 'その他' in x else 0)
    df['usage_office'] = df['用途'].apply(lambda x : 1 if '事務所' in x else 0)
    df['usage_home'] = df['用途'].apply(lambda x : 1 if '住宅' in x else 0)
    df['usage_shop'] = df['用途'].apply(lambda x : 1 if '店舗' in x else 0)
    df['usage_parking'] = df['用途'].apply(lambda x : 1 if '駐車' in x else 0)
    df['usage_share'] = df['用途'].apply(lambda x : 1 if '共同' in x else 0)
    df['usage_handwork'] = df['用途'].apply(lambda x : 1 if ('作業場' in x) |  ('倉庫' in x) | ('工場' in x) else 0)
    df.drop(columns=['用途'], inplace=True)
    return df

def feature_engineering(train, test):
    """
    Perform feature engineering
    """
    # deal timing
    train['deal_quater'] = train['取引時点'].apply(lambda x : torihiki_quater(x))
    test['deal_quater'] = test['取引時点'].apply(lambda x : torihiki_quater(x))
    train['deal_timing'] = train['取引時点'].apply(lambda x : jiten_convert(x))
    test['deal_timing'] = test['取引時点'].apply(lambda x : jiten_convert(x))
    
    train.drop(columns=['取引時点'], inplace=True)
    test.drop(columns=['取引時点'], inplace=True)
    
    # usage
    train = usage_feats(train)
    test = usage_feats(test)
    
    # kyori
    for f in ['延床面積（㎡）', '面積（㎡）', ]:
        train = change_to_number(train, f, f)
        test = change_to_number(test, f, f)    
    
    f = '最寄駅：距離（分）'
    train = change_to_minute(train, f, f)
    test = change_to_minute(test, f, f)
    
    for f in ['間口', '前面道路：幅員（ｍ）', '建ぺい率（％）', '容積率（％）']:
        train = change_to_float(train, f, f)
        test = change_to_float(test, f, f)

    train = madori_converter(train)
    test = madori_converter(test)
    madori_features = ['L', 'D', 'K', 'S', 'R', 'Maisonette', 'OpenFloor', 'Studio', 'RoomNum', 'Special', 'TotalRoomNum', 'RoomNumRatio']
    
    train['建築年'] = train['建築年'].apply(lambda x : kenchiku_convert(x))
    test['建築年'] = test['建築年'].apply(lambda x : kenchiku_convert(x))
    
    train['area_ratio2'] = train['面積（㎡）'] / (train['延床面積（㎡）'] + 1)
    test['area_ratio2'] = test['面積（㎡）'] / (test['延床面積（㎡）'] + 1)
    
    train['area_per_room'] = train['延床面積（㎡）'] / (train['TotalRoomNum'] + 1)
    test['area_per_room'] = test['延床面積（㎡）'] / (test['TotalRoomNum'] + 1)
    
    train['area_per_room2'] = train['面積（㎡）'] / (train['TotalRoomNum'] + 1)
    test['area_per_room2'] = test['面積（㎡）'] / (test['TotalRoomNum'] + 1)
    
    # numerical columns
    numerical_features = ['最寄駅：距離（分）', '面積（㎡）', '延床面積（㎡）', '前面道路：幅員（ｍ）', '建ぺい率（％）', '容積率（％）', '間口', '建築年', '取引時点', 
                          'area_ratio2', 'area_per_room', 'area_per_room2','deal_quater', 'deal_timing']

    # isnan as a feature
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    test_features = test.columns.values.tolist()
    y = train['y']
    df = pd.concat([train[test_features], test], ignore_index=True)
    df, isnan_features = onehot_nan(df, df.columns.values.tolist())
    train = df.iloc[:len(train)]
    train['y'] = y
    test = df.iloc[len(train):]
    
    # label encoding
    cat_features = [f for f in test_features if f not in drops + numerical_features + isnan_features + ['id', '間取り', '建築年', '取引時点', '市区町村コード', ] + madori_features]
    train, test, _ = label_encoding(train, test, cat_features)
    
    return train, test

train, test = feature_engineering(train, test)

### Translations
def english_columns(train, test):
    translation = {
         '種類': 'kinds',
         '地域': 'region',
         '市区町村コード': 'city_code',
         '地区名': 'area_name',
         '最寄駅：名称': 'nearest_station',
         '最寄駅：距離（分）': 'distance_to_station',
         '間取り': 'house_plan',
         '面積（㎡）': 'area',
         '土地の形状': 'shape',
         '間口': 'entry',
         '延床面積（㎡）': 'total_area',
         '建築年': 'built_year',
         '建物の構造': 'structure',
         '用途': 'usage',
         '今後の利用目的': 'purpose',
         '前面道路：方位': 'road_direction',
         '前面道路：種類': 'road_kinds',
         '前面道路：幅員（ｍ）': 'road_width',
         '都市計画': 'city_plan',
         '建ぺい率（％）': 'volume_area_ratio',
         '容積率（％）': 'total_area_ratio',
         '取引時点': 'deal_timing',
         '改装': 'renovation',
         '取引の事情等': 'issues',
    }
    for key in list(translation.keys()):
        translation[key + '_isnan'] = translation[key] + '_isnan'
    train = train.rename(columns=translation)
    test = test.rename(columns=translation)
    return train, test
train, test = english_columns(train, test)

### save
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
train.to_csv(output_path + 'train.csv', index=False)
test.to_csv(output_path + 'test.csv', index=False)
print('saved!')