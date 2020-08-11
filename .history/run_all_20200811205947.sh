#!/bin/sh
cd code

# feature engineering
python feature_engineering.py

# modeling
python modeling.py 'lgb' 3 217