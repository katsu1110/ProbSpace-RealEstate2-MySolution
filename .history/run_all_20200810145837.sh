#!/bin/sh
cd code

# feature engineering
python feature_engineering.py

# modeling
python modeling.py 'catb' 1 42
python modeling.py 'xgb' 2 1220
python modeling.py 'lgb' 3 217
python modeling.py 'nn' 4 116

# stacking
python run_stacking.py
