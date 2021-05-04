import os
import pandas as pd
import numpy as np

SEQ_LEN = 60 # dlzka sekvencie
FUTURE_PERIOD_PREDICT = 3 # kolko minut dopredu chceme predikovat rast/klesanie
PAIR_TO_PREDICT = 'LTC-USD' # kryptomena ktorej rast chceme predikovat

# klasifikacna funkcia pre rast/klesanie ceny
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

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