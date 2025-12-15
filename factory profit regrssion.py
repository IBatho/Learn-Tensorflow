import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

data_set = pd.read_csv("FU.csv")
print(data_set.head())

train_dataset = data_set.sample(frac=0.8, random_state=0)
test_dataset = data_set.drop(train_dataset.index)

print(sns.pairplot(train_dataset[["Profit", "Power", "Production Rate"]], diag_kind="kde"))