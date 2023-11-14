import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from anfis_ba import *


train_data = pd.read_csv("dataset/matlab_1_train.csv")

test_model = TSKModel()
test_model.create_rulebase_kmeans(train_data.iloc[:, 1:])
test_model.show_fuzzy_sets()