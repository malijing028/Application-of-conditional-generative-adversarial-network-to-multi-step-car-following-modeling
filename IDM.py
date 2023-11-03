import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import random

train_val_data = pd.read_csv('train_data.csv')
train_val_data_nan = pd.read_csv('train_data_nan.csv')
train_val_pairs = pd.read_csv('train_pairs.csv')
