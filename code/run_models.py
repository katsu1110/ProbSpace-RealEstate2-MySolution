"""
My quick & dirty modeling pipeline
"""

import numpy as np
import pandas as pd
import os
import sys
import gc
import re
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score

# model
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

# custom
from lgb_param_models import lgb_model
from xgb_param_models import xgb_model
from catb_param_models import catb_model
from lin_param_models import lin_model
from nn_param_models import nn_model
from get_pred import get_oof_ypred

class RunModel(object):
    """
    Model Fitting and Prediction Class:

    :INPUTS:

    :train_df: train pandas dataframe
    :test_df: test pandas dataframe
    :target: target column name (str)
    :features: list of feature names
    :categoricals: list of categorical feature names. Note that categoricals need to be in 'features'
    :model: 'lgb', 'xgb', 'catb', 'linear', or 'nn'
    :params: dictionary of hyperparameters. If empty dict {} is given, default hyperparams are used
    :task: 'regression', 'multiclass', or 'binary'
    :n_splits: K in KFold (default is 4)
    :cv_method: 'KFold', 'StratifiedKFold', 'TimeSeriesSplit', 'GroupKFold', 'StratifiedGroupKFold'
    :group: group feature name when GroupKFold or StratifiedGroupKFold are used
    :target_encoding: True or False
    :seed: seed (int)
    :scaler: None, 'MinMax', 'Standard'
    :verbose: bool

    :EXAMPLE:

    # fit LGB regression model
    model = RunModel(train_df, test_df, target, features, categoricals=categoricals,
            model="lgb", params={}, task="regression", n_splits=4, cv_method="KFold", 
            group=None, target_encoding=False, seed=1220, scaler=None)
    
    # save predictions on train, test data
    np.save("y_pred", model.y_pred)
    np.save("oof", model.oof)
    """

    def __init__(self, train_df : pd.DataFrame, test_df : pd.DataFrame, target : str, features : List, categoricals: List=[],
                model : str="lgb", params : Dict={}, task : str="regression", n_splits : int=4, cv_method : str="KFold", 
                group : str=None, target_encoding=False, seed : int=1220, scaler : str=None, verbose=True):

        # display info
        print("##############################")
        print(f"Starting training model {model} for a {task} task:")
        print(f"- train records: {len(train_df)}, test records: {len(test_df)}")
        print(f"- target column is {target}")
        print(f"- {len(features)} features with {len(categoricals)} categorical features")
        if target_encoding:
            print(f"- target encoding: Applied")
        else:
            print(f"- target encoding: NOT Applied")
        print(f"- CV strategy : {cv_method} with {n_splits} splits")
        if group is None:
            print(f"- no group parameter is used for validation")
        else:
            print(f"- {group} as group parameter")
        if scaler is None:
            print("- No scaler is used")
        else:
            print(f"- {scaler} scaler is used")
        print("##############################")

        # class initializing setups
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.features = features
        self.categoricals = categoricals
        self.model = model
        self.params = params
        self.task = task
        self.n_splits = n_splits
        self.cv_method = cv_method
        self.group = group
        self.target_encoding = target_encoding
        self.seed = seed
        self.scaler = scaler
        self.verbose = verbose
        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()

    def train_model(self, train_set, val_set):
        """
        employ a model
        """
        # compile model
        if self.model == "lgb": # LGB             
            model, fi = lgb_model(self, train_set, val_set)

        elif self.model == "xgb": # xgb
            model, fi = xgb_model(self, train_set, val_set)

        elif self.model == "catb": # catboost
            model, fi = catb_model(self, train_set, val_set)

        elif self.model == "linear": # linear model
            model, fi = lin_model(self, train_set, val_set)

        elif self.model == "nn": # neural network
            model, fi = nn_model(self, train_set, val_set)
        
        return model, fi # fitted model and feature importance

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        """
        dataset converter
        """
        if (self.model == "lgb") & (self.task != "multiclass"):
            train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
            val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
            
        elif (self.model == "nn") & (self.task == "multiclass"):
            ohe = OneHotEncoder(sparse=False, categories='auto')
            train_set = {'X': x_train, 'y': ohe.fit_transform(y_train.values.reshape(-1, 1))}
            val_set = {'X': x_val, 'y': ohe.transform(y_val.values.reshape(-1, 1))}
            
        else:
            train_set = {'X': x_train, 'y': y_train}
            val_set = {'X': x_val, 'y': y_val}
            
        return train_set, val_set

    def calc_metric(self, y_true, y_pred): 
        """
        calculate evaluation metric for each task
        this may need to be changed based on the metric of interest
        """
        if self.task == "multiclass":
            return f1_score(y_true, y_pred, average="macro")
        
        elif self.task == "binary":
            return roc_auc_score(y_true, y_pred) # log_loss
        
        elif self.task == "regression":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def get_cv(self):
        """
        employ CV strategy
        """

        # return cv.split
        if self.cv_method == "KFold":
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df)
        
        elif self.cv_method == "StratifiedKFold":
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            return cv.split(self.train_df, self.train_df[self.target])
        
        elif self.cv_method == "TimeSeriesSplit":
            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)
            return cv.split(self.train_df)
        
        # elif self.cv_method == "GroupKFold":
        #     cv = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        #     return cv.split(self.train_df, self.train_df[self.target], self.group)
        
        # elif self.cv_method == "StratifiedGroupKFold":
        #     cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        #     return cv.split(self.train_df, self.train_df[self.target], self.group)

    def fit(self):
        """
        perform model fitting        
        """

        # initialize
        y_vals = np.zeros((self.train_df.shape[0], ))
        if self.task == "multiclass":
            n_class = len(np.unique(self.train_df[self.target].values))
            oof_pred = np.zeros((self.train_df.shape[0], n_class))
            y_pred = np.zeros((self.test_df.shape[0], n_class))
        else:
            oof_pred = np.zeros((self.train_df.shape[0], ))
            y_pred = np.zeros((self.test_df.shape[0], ))

        # group does not kick in when group k fold is used
        if self.group is not None:
            if self.group in self.features:
                self.features.remove(self.group)
            if self.group in self.categoricals:
                self.categoricals.remove(self.group)
        fi = np.zeros((self.n_splits, len(self.features)))

        # target encoding
        numerical_features = [f for f in self.features if f not in self.categoricals]
        if self.target_encoding:            
            # perform target encoding
            for c in self.categoricals:
                data_tmp = pd.DataFrame({c: self.train_df[c].values, 'target': self.train_df[self.target].values})
                tmp = np.nan * np.ones(self.train_df.shape[0])
                
                cv = self.get_cv()
                for fold, (train_idx, val_idx) in enumerate(cv):
                    target_mean = data_tmp.iloc[train_idx].groupby(c)['target'].mean()
                    tmp[val_idx] = self.train_df[c].iloc[val_idx].map(target_mean).values
                self.train_df[c] = tmp
                
                # replace categorical variable in test
                target_mean = data_tmp.groupby(c)['target'].mean()
                self.test_df.loc[:, c] = self.test_df[c].map(target_mean).values
            
            # no categoricals any more
            numerical_features = self.features.copy()
            self.categoricals = []
        
        # fill nan
        if self.model not in ['lgb', 'catb', 'xgb']:
            # fill NaN (numerical features -> median, categorical features -> mode)
            self.train_df[numerical_features] = self.train_df[numerical_features].replace([np.inf, -np.inf], np.nan)
            self.test_df[numerical_features] = self.test_df[numerical_features].replace([np.inf, -np.inf], np.nan)
            self.train_df[numerical_features] = self.train_df[numerical_features].fillna(self.train_df[numerical_features].median())
            self.test_df[numerical_features] = self.test_df[numerical_features].fillna(self.test_df[numerical_features].median())
            self.train_df[self.categoricals] = self.train_df[self.categoricals].fillna(self.train_df[self.categoricals].mode().iloc[0])
            self.test_df[self.categoricals] = self.test_df[self.categoricals].fillna(self.test_df[self.categoricals].mode().iloc[0])
      
        # scaling, if necessary
        if self.scaler is not None:
            # to normal
            pt = QuantileTransformer(n_quantiles=100, random_state=self.seed, output_distribution="normal")
            self.train_df[numerical_features] = pt.fit_transform(self.train_df[numerical_features])
            self.test_df[numerical_features] = pt.transform(self.test_df[numerical_features])

            # starndardize
            if self.scaler == "MinMax":
                scaler = MinMaxScaler()
            elif self.scaler == "Standard":
                scaler = StandardScaler()
            self.train_df[numerical_features] = scaler.fit_transform(self.train_df[numerical_features])
            self.test_df[numerical_features] = scaler.transform(self.test_df[numerical_features])

            x_test = self.test_df.copy()
            if self.model == "nn":
                x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]
            else:
                x_test = x_test[self.features]
        else:
            x_test = self.test_df[self.features]
        
        # fitting with out of fold
        cv = self.get_cv()
        for fold, (train_idx, val_idx) in enumerate(cv):
            # train test split
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target].iloc[train_idx], self.train_df[self.target].iloc[val_idx]

            if self.model == "nn":
                x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]
                x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]

            # model fitting
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model, importance = self.train_model(train_set, val_set)
            fi[fold, :] = importance
            y_vals[val_idx] = y_val

            # predictions and check cv score
            oofs, ypred = get_oof_ypred(model, x_val, x_test, self.model, self.task)
            y_pred += ypred.reshape(y_pred.shape) / self.n_splits
            if self.task == "multiclass":
                oof_pred[val_idx, :] = oofs.reshape(oof_pred[val_idx, :].shape)
                print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_vals[val_idx], 
                    np.argmax(oof_pred[val_idx, :], axis=1))))
            else:
                oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)
                print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_vals[val_idx], 
                    oof_pred[val_idx])))

        # feature importance data frame
        fi_df = pd.DataFrame()
        for n in np.arange(self.n_splits):
            tmp = pd.DataFrame()
            tmp["features"] = self.features
            tmp["importance"] = fi[n, :]
            tmp["fold"] = n
            fi_df = pd.concat([fi_df, tmp], ignore_index=True)
        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()
        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))

        # outputs
        if self.task == "multiclass":
            loss_score = self.calc_metric(y_vals, np.argmax(oof_pred, axis=1))
        else:
            loss_score = self.calc_metric(y_vals, oof_pred)

        if self.verbose:
            print('Our oof loss score is: ', loss_score)
        return y_pred, loss_score, model, oof_pred, y_vals, fi_df

    def plot_feature_importance(self, rank_range=[1, 50]):
        """
        function for plotting feature importance (nothing is returned when the model is NN)

        :EXAMPLE:
        # fit LGB regression model
        model = RunModel(train_df, test_df, target, features, categoricals=categoricals,
                model="lgb", task="regression", n_splits=4, cv_method="KFold", 
                group=None, seed=1220, scaler=None)
        
        # plot 
        fi_df = model.plot_feature_importance(rank_range=[1, 100])
        
        """
        # plot feature importance
        _, ax = plt.subplots(1, 1, figsize=(10, 20))
        sorted_df = self.fi_df.sort_values(by = "importance_mean", ascending=False).reset_index().iloc[self.n_splits * (rank_range[0]-1) : self.n_splits * rank_range[1]]
        sns.barplot(data=sorted_df, x ="importance", y ="features", orient='h')
        ax.set_xlabel("feature importance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return sorted_df