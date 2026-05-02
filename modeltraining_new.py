#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm')


# In[1]:


import pandas as pd
import numpy as np
import pickle
import warnings
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, normalize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    VotingRegressor, StackingRegressor,
)
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    classification_report, accuracy_score, silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor, XGBClassifier

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
    print("LightGBM available.")
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not installed. Run: pip install lightgbm")

warnings.filterwarnings('ignore')
os.makedirs("artifacts", exist_ok=True)
print("Imports done.")

