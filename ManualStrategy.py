# df_trades = ms.testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000) 
"""
Create ManualStrategy.py and implement a set of rules using at a minimum of 3 indicators
 you created in Project 6 (NOTE: You can make changes to the indicators to properly work
  with both Manual Strategy and Strategy Learner but both strategies must use the same indicator code). 
  Devise some simple logic using your indicators to enter and exit positions in the stock. 
  All indicators must be used in some way to determine a buy/sell signal.  You cannot use a single indicator 
  for all signals. 

A recommended approach is to create a single logical expression that yields a -1, 0, or 1, 
corresponding to a “short,” “out” or “long” position. Example usage is signal: 
If you are out of the stock, then a 1 would signal a BUY 1000 order. 
If you are long, a -1 would signal a SELL 2000 order. 
You don’t have to follow this advice though, so long as you follow the trading rules outlined above. 

For the report we want a written description, not code, however, it is OK to augment your written 
description with a pseudocode figure. 

You should tweak your rules as best you can to get the best performance possible during 
the in-sample period (do not peek at out-of-sample performance) and 
should include more than one trade.  Use your rule-based strategy to generate a trades DataFrame 
over the in-sample period. 

We expect that your rule-based strategy should outperform the benchmark over the in-sample period. 

Benchmark: The performance of a portfolio starting with $100,000 cash, 
investing in 1000 shares of JPM on the first trading day, and holding that position. 
"""

import datetime as dt  		
from datetime import timedelta  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
from numpy.lib.shape_base import expand_dims
from numpy.testing._private.utils import decorate_methods  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
from marketsimcode import *
import indicators
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def author(): 
  return 'sawid3' # replace tb34 with your Georgia Tech username.  

def testPolicy( symbol = "JPM", 
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009,12,31), 
    sv = 100000): 		  	   		   	 		  		  		    	 		 		   		 		   		  	   		   	 		  		  		    	 		 		   		 		  
    """ this function generates the df that corresponds to daily trades based 
    on seeing future prices to generate max possible return"""
    ############# Prepare Data!!
    #########################	CALL DF PRICES
    prices=get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    volume=get_data([symbol], pd.date_range(sd, ed),colname="Volume", addSPY=False)
    volume.dropna(inplace=True)
    volume=volume/volume.iloc[0]
    prices.dropna(inplace=True)
    prices=prices/prices.iloc[0]
    # prices=prices.fillna(method='ffill')
    # prices=prices.fillna(method='bfill')
    # print(prices)
    # print("============================================")
    # holdings=0
    trades=prices[symbol].copy()
    trades[:]=0
    dates=trades.index
    sma=indicators.sma(prices,period=50)
    ema=indicators.ema(prices,period=20)
    BBP=indicators.BBP(prices,period=50)
    mom=indicators.mom(prices,period=50)
    obv=indicators.obv(prices, volume)
    recorded_holding=trades.copy()
    holding=0
    for i in trades.index:
      # buy
      if  (mom.loc[i]<-0.05).bool() and (BBP.loc[i]<0).bool() and (sma.loc[i]<0.95).bool():
        if holding==0:
          order=1000
          holding=holding+order #1000

        elif holding==-1000:
          order=2000
          holding=holding+order #1000
        else:
          order=0
          holding=holding+order

      elif (mom.loc[i]>0.05).bool() and (BBP.loc[i]>1).bool() and (sma.loc[i]>1.05).bool() :
        if holding==0:
          order=-1000
          holding=holding+order #-1000
        elif holding ==1000:
          order=-2000
          holding=holding+order #-1000
        else:
          order=0
          holding=holding+order
      else:
        order=0
        holding=holding+order
      
      # print(holding)

      recorded_holding.loc[i]=holding
      trades.loc[i]=order

        # elif holding==100
    # print(trades)
    # def buy_position(holdings):
    #   trade=-holdings+1000 # this to limit the purchase of 2000 s.t. holdings between -1000,0,1000

    #   return trade
    # def sell_position(holdings):
    #   trade=-holdings-1000 # this to limit the purchase of 2000 s.t. holdings between -1000,0,1000

    #   return trade

    # for day in range(len(trades)-1):
    #     if  (mom.loc[i]<-0.05).bool() and (BBP.loc[i]<0.2).bool()  and (holding<1000): # if price goes up buy, else sell or do nothing

    #       trades[day]=buy_position(holdings)
    #     else: 
    #       trades[day]= sell_position(holdings)

    #     holdings+=trades[day]


    trades=pd.DataFrame(trades)
    # print( trades.loc[trades!=0])
    test=trades

  	 		  		  		    	 		 		   		 		  
    return trades  

if __name__ == "__main__":  
    test=testPolicy( symbol = "JPM", 
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009,12,31), 
    sv = 100000)
    # print(test)
    val=compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    test,	  	   		   	 		  		  		    	 		 		   		 		  
    start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005,)
    df_bench=gen_benchmark(symbol='JPM',sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009,12,31), sv = 100000, commission=9.95, impact=0.005)
    # df_port_val=val/val.iloc[0]
    # df_bench=df_bench/df_bench.iloc[0]

    # tos.gen_plots(df_port_val,df_bench)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(val)		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_stats(df_bench)
      
 		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {val[-1]}")  
    print(f"Final Benchmark Portfolio Value: {df_bench[-1]}")  		  	 		  	 