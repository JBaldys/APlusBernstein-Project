import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from src.constants import model_data_dir

df_train = pd.read_csv(model_data_dir + "/train_classification.csv")

model = Sequential()
model.add(LSTM(64, input_shape=()))
