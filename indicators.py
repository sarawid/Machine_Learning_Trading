
# list:
# ema
# %bb*
# mom
# sma/price*

import datetime as dt  		
from datetime import timedelta  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
from numpy.lib.shape_base import expand_dims
from numpy.testing._private.utils import decorate_methods  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def author(): 
  return 'sawid3' # replace tb34 with your Georgia Tech username.  

def normalize_df(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df

def sma(df, period,standard=False):
    """code obtained from Vectorize me PPT"""
    sma = df.rolling(period).mean()
    # for day in range(period, df.shape[0]):
    #     sma.iloc[day,:]=df.iloc[day,:]/sma.iloc[day,:]
    return sma
def ema(df, period, standard=False):
    """Apply  exponential weighted  function to get weighted means"""
    ema= df.ewm(span=period, min_periods=period).mean()
    return ema

def BBP(df, period, standard=False):
    """code obtained from Vectorize me PPT """
    rolling_std=df.rolling(window=period).std()
    sma=df.rolling(period).mean()
    top_band=sma+(2*rolling_std)
    bottom_band=sma-(2*rolling_std)
    bbp=(df-bottom_band)/(top_band-bottom_band)
    return bbp

def mom(df, period, standard=False):
    """rate of change of price"""
    mom=(df/df.shift(period)) -1
    return mom

def obv(close, volume,standard=False):
    """If the closing price is above the prior close price then: 
        Current OBV = Previous OBV + Current Volume

        If the closing price is below the prior close price then: 
        Current OBV = Previous OBV  -  Current Volume

        If the closing prices equals the prior close price then:
        Current OBV = Previous OBV (no change)"""
        
    obv=np.where(close> close.shift(1), volume, 
    np.where(close < close.shift(1), -volume, 0)).cumsum()
    obv=pd.DataFrame(data=obv, index=volume.index, columns=volume.columns)
    # pd.DataFrame()
    return obv

# def main():
    
if __name__ == "__main__":
    plt.close('all')
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2009,12,31)
    JPM=get_data(['JPM'], pd.date_range(sd, ed), addSPY=False)
    JPM.dropna(inplace=True)
    JPM=JPM/JPM.iloc[0]
    JPV=get_data(['JPM'], pd.date_range(sd, ed), colname="Volume", addSPY=False)
    JPV.dropna(inplace=True)
    JPV=JPV/JPV.iloc[0]
    sma=sma(JPM, period=50)
    ema=ema(JPM,period=50)
    BBP=BBP(JPM,period=50)
    mom=mom(JPM,period=50)
    obv=obv(JPM, JPV)
    # test
    fig, ax = plt.subplots()
    plt.title("JPM Price and SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.plot(JPM, label='Price', color='green')
    plt.plot(sma,label='SMA', color='red')
    fig.autofmt_xdate()  
    plt.legend()
    plt.savefig("SMA.png")


    plt.close('all')
    fig, ax = plt.subplots()
    plt.title("JPM Price and EMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.plot(JPM, label='Price', color='green')
    plt.plot(ema,label='EMA', color='red')
    fig.autofmt_xdate()  
    plt.legend()
    plt.savefig("EMA.png")

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle("JPM Price and Bollinger Band %  ")
    # ax2.set_title("On Balance Volume ")
    ax1.set_title("JPM Price ")
    ax2.set_title(" Bollinger Band % ")

    ax1.set(xlabel="Date",ylabel="Price")
    ax2.set(xlabel="Date",ylabel="BBP")
    ax1.plot(JPM, label='Price', color='green')
    ax2.plot(BBP,label='BBP', color='red')
    fig.autofmt_xdate()  
    plt.legend()
    plt.savefig("BBP.png")


    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2,sharex=True)
    fig.suptitle("JPM Price and Momentum ")
    ax1.set_title("JPM Price ")
    ax2.set_title("Momentum ")
    # ax2.set_title("On Balance Volume ")

    ax1.set(xlabel="Date",ylabel="Price")
    ax2.set(xlabel="Date",ylabel="Price Momentum")

    ax1.plot(JPM, label='Price', color='green')
    ax2.plot(mom,label='Momentum', color='red')
    fig.autofmt_xdate()  
    plt.legend()
    plt.savefig("mom.png")

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("JPM Price and On Balance Volume ")

    ax1.set_title("JPM Price ")
    ax2.set_title("On Balance Volume ")

    ax1.set(xlabel="Date",ylabel="Price")
    ax2.set(xlabel="Date",ylabel="Volume")

    # ax2.xlabel("Date")
    # ax1.ylabel("Volume")
    # ax2.ylabel("Volume")

    ax1.plot(JPM, label='Price', color='green')
    ax2.plot(obv,label='On Balance Volume', color='red')
    fig.autofmt_xdate()  
    plt.savefig("obv.png")
