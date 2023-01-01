""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		
from datetime import timedelta  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
from numpy.lib.shape_base import expand_dims  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  


def author(): 
  return 'sawid3' # replace tb34 with your Georgia Tech username.  

def compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    df_trades,	  	   		   	 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		   	 		  		  		    	 		 		   		 		  
):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		   	 		  		  		    	 		 		   		 		  
    :type df_trades: dataframe of trades indexed by dates 		  	   		   	 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		  	   		   	 		  		  		    	 		 		   		 		  
    # TODO: Your code here  		  
    # ############ Prepare Data!!
     # think about how you can return a dataframe as well with columns that are the same as order_file
     # generate date, stock, shares, order df
    df=df_trades
    symbol=df.columns[0]
    # df.columns='Shares'

    start_date=min(df.index)
    end_date=max(df.index)

    	    	 		 		   		 		  
  	#########################	CALL DF PRICES
    prices=get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    prices = prices.assign(Cash=1.0)
    prices.dropna(inplace=True)
    # print(prices)
    # print("============================================")


      
    ########################### CALL DF TRADES: Date Stocks and Cash --> how many stockas were purchased in x day for aapl

    df2 = pd.DataFrame(np.zeros(prices.shape))
    df2.index = prices.index
    df2.columns = prices.columns
    dates=df.index
    # df2['Transaction_cost']=0.0
    for day in range(len(df)):
        # ix=str(dates[day])[0:10]
        # if df.loc[row,'Date'] in df2.index: #  check if it is a valid trade day
        order=df.loc[dates[day],symbol] # get date to index in prices

        df2.loc[dates[day],symbol]=df.loc[dates[day],symbol] 
        if order>0:
            
            cost=prices.loc[dates[day],symbol]*(1+impact)*order*(-1)
            df2.loc[dates[day],'Cash']=-(commission)+cost

        elif order <0 :
            cost=prices.loc[dates[day],symbol]*(1-impact)*order*(-1) #cost needs to be positive to cancel out the negative of the shares
            df2.loc[dates[day],'Cash']=cost-(commission)
        elif order ==0:
            df2.loc[dates[day],'Cash']=0

    trades=df2
    ####################################### HOLDINGS ###################################
    holdings= pd.DataFrame(np.zeros(prices.shape))
    holdings.index = prices.index
    holdings.columns = prices.columns
    #initialize cash in the first row
    initial = pd.to_datetime(start_date) - timedelta(days=1)
    # initial = initial.strftime("%Y-%m-%d")
    holdings.loc[initial]=0
    holdings.loc[initial,'Cash']=start_val
    holdings.reset_index()
    holdings.sort_index(inplace=True)



    for i in range(len(holdings.index)-1):
        r1=(holdings.index)[i]
        r2=(holdings.index)[i+1]
        # date_1=str(r+1)[0:10]
        if  r2 in holdings.index:
            holdings.loc[r2]=holdings.loc[r1]+trades.loc[r2]
    # print(holdings)
    # print('#######################################################################')


    Values=prices.multiply(holdings,axis=0)
    Values['Cash']=holdings['Cash']
    Values['Total']=Values.sum(axis=1)
    Values=Values.loc[start_date:end_date]
    portvals=Values['Total']
    # print(Values)
    # print(Values)



    ######################### 	   		   	 		  		  		    	 		 		   		 		  
    return portvals  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def gen_benchmark(symbol, sd, ed, 
  sv=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005 ):
  """this function generates  benchmark portfolio values based on holding JPM
  Benchmark: The performance of a portfolio starting with $100,000 cash,
    investing in 1000 shares of JPM, and holding that position"""
  prices=get_data([symbol], pd.date_range(sd, ed), addSPY=False)
  prices.dropna(inplace=True)
  indx=[prices.index[0], prices.index[len(prices.index) - 1]]
  df_trades = pd.DataFrame(data=[1000,0], index=indx, columns=[symbol])
  # df_trades.index=prices.index
  # df_trades.columns=prices.columns
  # df_trades.loc[sd]=1000
  

  port_df=compute_portvals(df_trades,  
    start_val=sv,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=commission,  		  

    impact=impact )
  return port_df 	 		  		  		    	 		 		   		 		  
                                                            
def gen_plots(tos_df, df_bench ):
  fig, ax = plt.subplots()
  plt.title("Manual Strategy Portfolio Values: Manual vs. Benchmark")
  plt.xlabel("Date")
  plt.ylabel("Portfolio Value")
  plt.plot(tos_df, label='Manual', color='red')
  plt.plot(df_bench,label='Benchmark', color='green')
  # formatter = mdates.DateFormatter("%Y-%m-%d")
  fig.autofmt_xdate()  
  plt.legend()
  plt.savefig("Figure_1.png")
  # plt.show()


def port_stats(port_val):
    # code obtained from class notes in lesson 01-04 and 01-07
    # print(type(prices))
    k=252
    daily_rf=0.0
    # port_val=port_value(prices,allocs)
    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1

    daily_returns = port_val.copy()
    daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans

    cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    adr = daily_returns[1:].mean()
    sddr = daily_returns[1:].std()
    sr = np.sqrt(k) * np.mean(daily_returns[1:] - daily_rf) /sddr
    return  cr, adr, sddr, sr   		  	   		   	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		   	 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # of = "./additional_orders/orders2.csv"  	
    # df=pd.read_csv(of)
    # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # df.sort_values(by = 'Date')
    # df.dtypes
    # sd=min(df['Date'])
    # ed=max(df['Date'])	  	   		   	 		  		  		    	 		 		   		 		  
    # sv = 1000000
    # dates = pd.date_range(sd, ed)  		  	   		   	 		  		  		    	 		 		   		 		  
    # SPY = get_data(["SPY"], dates)	   	 		  		  		    	 		 		   		 		  
    # # Process orders  		  	   		   	 		  		  		    	 		 		   		 		  
    # portvals = compute_portvals(orders_file=of, start_val=sv,  commission = 9.95, impact = 0.005)  		  	   		   	 		  		  		    	 		 		   		 		  
    # if isinstance(portvals, pd.DataFrame):  		  	   		   	 		  		  		    	 		 		   		 		  
    #     portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		   	 		  		  		    	 		 		   		 		  
    # else:  		  	   		   	 		  		  		    	 		 		   		 		  
    #     "warning, code did not return a DataFrame"  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # # Get portfolio stats  		  	   		   	 		  		  		    	 		 		   		 		  
    # # Here we just fake the data. you should use your code from previous assignments.  		  	   		   	 		  		  		    	 		 		   		 		  
    # # start_date = dt.datetime(2008, 1, 1)  		  	   		   	 		  		  		    	 		 		   		 		  
    # # end_date = dt.datetime(2008, 6, 1)  	
    # def port_stats(port_val):
    #     # code obtained from class notes in lesson 01-04 and 01-07
    #     # print(type(prices))
    #     k=252
    #     daily_rf=0.0
    #     # port_val=port_value(prices,allocs)
    #     cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1

    #     daily_returns = port_val.copy()
    #     daily_returns[1:] = (port_val[1:] / port_val[:-1].values) - 1
    #     daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans

    #     cr = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    #     adr = daily_returns[1:].mean()
    #     sddr = daily_returns[1:].std()
    #     sr = np.sqrt(k) * np.mean(daily_returns[1:] - daily_rf) /sddr
    #     return  cr, adr, sddr, sr 

    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(portvals)		 		   		 		  
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_stats(SPY)		 		  
 		    	 		 		   		 		  
    # # Compare portfolio against $SPX  		  	   		   	 		  		  		    	 		 		   		 		  
    # # print(f"Date Range: {start_date} to {end_date}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
#     port_val=test_code()  		  	   		   	 		  		  		    	 		 		   		 		  
#     # print(port_val)