import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from pylab import rcParams
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

CRYPTO_TO_PREDICT = 'Bitcoin'
RANDOM_SEED = 42
DROPOUT = 0.2
SEQ_LEN = 100
BATCH_SIZE = 64
WINDOW_SIZE = SEQ_LEN - 1
EPOCHS = 50
NAME = f'{int(time.time())}_{SEQ_LEN}__{CRYPTO_TO_PREDICT}_model'

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)), )

# dataset https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory
crypto_dataset = os.path.join(__location__, f'crypto_data/coin_{CRYPTO_TO_PREDICT}.csv') # bitcoin dataset
logs_folder = os.path.join(__location__, 'logs')
models_folder = os.path.join(__location__, 'models')

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

np.random.seed(RANDOM_SEED)

# normalizacia dat
def normalize(df):
    scaler = MinMaxScaler()

    close_price = df.Close.values.reshape(-1, 1)

    scaled_close = scaler.fit_transform(close_price)
    print("\nscaled_close.shape", scaled_close.shape)

    #print(np.isnan(scaled_close).any())

    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)

    #print(np.isnan(scaled_close).any())

    return scaled_close, scaler

# rozdelenie na sekvencie
def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

# preprocessing dat
def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test

df = pd.read_csv(crypto_dataset, parse_dates=['Date']) # nacitanie datasetu

df = df.sort_values('Date')

print( df.head() )

print( "\ndf.shape" ,df.shape )

# zobrazenie datasetu
ax = df.plot(x='Date', y='Close')
ax.set_xlabel("Date")
ax.set_ylabel(f" {CRYPTO_TO_PREDICT} - Close Price (USD)")
plt.show()

# normalizacia
scaled_close, scaler = normalize(df)

# preprocesing, poslednych 5% su testovacie data
X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95) 

print( "\nX_train.shape", X_train.shape)

print( "\nX_test.shape", X_test.shape)

# vytvorenie modelu
model = keras.Sequential() # sekvencny model

model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, X_train.shape[-1]))) # pridanie LSTM, 'tanh' je aktivacna funkcia pre CuDNNLSTM
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=False)))

# vystupna vrstva
model.add(Dense(units=1)) # 1 vystup , predikcia ceny
model.add(Activation('linear')) # linearna aktivacna funkcia pre vystupnu vrstvu

# komplilacia modelu
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

tensorboard = TensorBoard(log_dir= f"{logs_folder}/{NAME}" ) # sledovanie modelu v realnom case, tensorboard --logdir=logs/

# trenovanie
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_split=0.1,
    callbacks=[tensorboard]
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Chyby')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# print( '\nhistory keys:', history.history.keys())

# predikcia
y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Realna Cena", color='green')
plt.plot(y_hat_inverse, label="Predpovedana Cena", color='red')
 
plt.title(f'{CRYPTO_TO_PREDICT} - predikcia ceny')
plt.xlabel('Cas [den]')
plt.ylabel('Cena')
plt.legend(loc='best')

plt.show()