import os
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from collections import deque

SEQ_LEN = 60 # dlzka sekvencie
FUTURE_PERIOD_PREDICT = 3 # kolko minut dopredu chceme predikovat rast/klesanie
PAIR_TO_PREDICT = 'LTC-USD' # kryptomena ktorej rast chceme predikovat

# klasifikacna funkcia pre rast/klesanie ceny
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# Preprocesing a sekvencie
def preprocess_df(df):
    df = df.drop( f"{PAIR_TO_PREDICT}_future", 1 ) # odstranenie future stlpca - sluzil len pre vytvorenie target stlpca

    for col in df.columns:
        if col != f"{PAIR_TO_PREDICT}_target":
            df[col] = df[col].pct_change() # normalizacia hodnot na percenta
            df.dropna(inplace=True) # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values) # skalovanie medzi 0 a 1
    
    df.dropna(inplace=True) # pri normalizacii mohly vzniknut prazdne hodnoty

    sequential_data = []  # list pre ukladanie sekvencnych dat
    prev_days = deque(maxlen=SEQ_LEN)  # sekvencie

    # print(df.values)

    for i in df.values:  # iterovanie riadkami
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append feature and labels sequences

    random.shuffle(sequential_data)  # pomiesanie dat

    buys = []  # sekvencia nakupovacich - buy hodnot
    sells = []  # sekvencia predavacich - sel hodnot

    for seq, target in sequential_data:  # iterovanie sekvencnymi datami
        if target == 0: 
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target]) 

    random.shuffle(buys)
    random.shuffle(sells)

    # balancovanie buy/sell hodnot
    lower = min(len(buys), len(sells)) 
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells  # spojenie do sekvencie
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)  # sekvencia
        y.append(target)  # nazvy - buy/sell

    return np.array(X), y

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)), )

data_folder = os.path.join(__location__, 'crypto_data') # priecinok s datasetmi

master_df = pd.DataFrame() # hlavny df

pairs = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD'] # pary ktore budeme analyzovat

for pair in pairs:
    dataset = f"{data_folder}/{pair}.csv" # cela cesta k suboru

    df = pd.read_csv(dataset) # names = ["time", "low", "high", "open", "close", "volume"]
    #print( 'Crypro pair : \n\n' , pair ,df.head())

    # premenovanie stlpcov
    df.rename( columns={"Close" : f"{pair}_close", "Volume" : f"{pair}_volume"} , inplace=True )

    df.set_index("Unix", inplace=True) # unix timestamp je index

    df = df[ [f"{pair}_close", f"{pair}_volume"] ] # zaujma nas iba close a volume

    # print(df.head())

    # mergovanie do hlavneho datasetu
    if len(master_df) == 0: 
        master_df = df 
    else:
        master_df =  master_df.merge(df,left_index=True,right_index=True) 

master_df.fillna(method="ffill", inplace=True)  # ak existuju medzery v udajoch, pouzije sa predchodzi udaj
master_df.dropna(inplace=True) # odstranenie na hodnot

# print( master_df.head() )

master_df[f"{PAIR_TO_PREDICT}_future"] = master_df[f"{PAIR_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) # vytorenie future stlpca na zaklade velkosti predikcie

master_df[f"{PAIR_TO_PREDICT}_target"] = list(map(classify, master_df[f"{PAIR_TO_PREDICT}_close"], master_df[f"{PAIR_TO_PREDICT}_future"])) # vytvorenie target stlpca

# print( master_df[[f"{PAIR_TO_PREDICT}_close", f"{PAIR_TO_PREDICT}_future", f"{PAIR_TO_PREDICT}_target"]].head(10) )

times = sorted(master_df.index.values)
last_5pct = sorted(master_df.index.values)[-int(0.05*len(times))]  # poslednych 5% vsetkych dat na verifikaciu

# print( last_5pct )

validation_master_df = master_df[(master_df.index >= last_5pct)]  # validacne data budu poslednych 5%
master_df = master_df[(master_df.index < last_5pct)]  # odstranenie poslednych 5% ktore su vo validacii

train_x, train_y = preprocess_df(master_df)
validation_x, validation_y = preprocess_df(validation_master_df)

print(f"Trenovacie data: {len(train_x)} Validacne data: {len(validation_x)}")
print(f"Predaje: {train_y.count(0)}, Nakupy: {train_y.count(1)}")
print(f"Validacia predaje: {validation_y.count(0)}, Validacia nakupy: {validation_y.count(1)}")