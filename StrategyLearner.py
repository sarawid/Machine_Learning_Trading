""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
import random  
import math		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
import util as ut  		
import numpy as np
import BagLearner  as bl
import RTLearner as rt
import indicators
from marketsimcode import *
# import numpy as np
def author(): 
  return 'sawid3'
  		  	   		   	 		  		  		    	 		 		   		 		  	  	   		   	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission 
        # self.N=N 		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  	
    	  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """ 
        #set seed in RTlearner 		  	   		   	 		  		  		    	 		 		   		 		  
        N=30
        leaf_size=5
        bags=50 
        YBUY=0.001
        YSELL=0.001 		   	 		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		   	 		  		  		    	 		 		   		 		  
        prices=ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False) 
        prices.dropna(inplace=True)
        prices2=prices/prices.iloc[0] 
        sma=indicators.sma(prices2,period=50)
        BBP=indicators.BBP(prices2,period=50)
        mom=indicators.mom(prices2,period=50)
        x=pd.concat([sma,BBP,mom], axis=1)
        x.dropna(inplace=True)
        x = x[:-N]
        x = x.values

        # N=24
        Y=np.zeros((len(x),1))
        longs=[]
        shorts=[]
        for i in range(len(x)-1):
            t=prices.index[i+50]
            tN=prices.index[i+50+N]
            delta=(abs(prices.loc[tN]-prices.loc[t])-self.commission).values[0]
            ret=(((prices.loc[tN]-prices.loc[t])/prices.loc[t])).values[0]
            long = (((prices.loc[tN]-self.commission-prices.loc[t])/prices.loc[t])).values[0] # return on long
            longs.append(long)
            short=(((prices.loc[t]-self.commission-prices.loc[tN])/prices.loc[t])).values[0] # return on short
            shorts.append(short)
            if ret > YBUY+self.impact and delta>0:
                    Y[i] = +1 # LONG
            elif ret < YSELL-self.impact and delta >0 :
                    Y[i] = -1 # SHORT
            else:
                    Y[i] = 0 # CASH
        Y=Y
        # print(longs, shorts)
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size':leaf_size}, bags=bags, boost=False, verbose=False)
        self.learner.add_evidence(x, Y)
        # return Y

 		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
 		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		   	 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		  	   		   	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data 
        # N=15
        # YBUY=0.001
        # YSELL=0.001
        prices=ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False) 
        prices.dropna(inplace=True)
        prices2=prices/prices.iloc[0] 

        trades=prices[symbol].copy()
        trades[:]=0


        sma=indicators.sma(prices2,period=50)
        BBP=indicators.BBP(prices2,period=50)
        mom=indicators.mom(prices2,period=50)
        x=pd.concat([sma,BBP,mom], axis=1)
        # x.dropna(inplace=True)
        # x = x[:-N]
        x = x.values
        
        pred=self.learner.query(x)	
        Y_pred=np.zeros((len(x),1))
        for i in range(len(x)-1):
            if pred[0][i] >= 0.5:
                Y_pred[i]=1
            elif pred[0][i] <= -0.5:
                Y_pred[i]= -1
            else:
                Y_pred[i] = 0

        # actual=
        # Y_test=np.zeros((len(x),1))


        # for i in range(len(x)-1):
        #     t=prices.index[i]
        #     tN=prices.index[i+N]
        #     long = (((prices.loc[tN]-self.commission-prices.loc[t])/prices.loc[t])).values[0] # return on long
        #     short=(((prices.loc[t]-self.commission-prices.loc[tN])/prices.loc[t])).values[0] # return on short
        #     if long > YBUY+self.impact:
        #             Y_test[i] = +1 # LONG
        #     elif short > YSELL+self.impact:
        #             Y_test[i] = -1 # SHORT
        #     else:
        #             Y_test[i] = 0 # CASH
        
        holding=0	  
        for i in range(len(Y_pred)):
            if  Y_pred[i]==1:
                if holding==0:
                    order=1000
                    holding=holding+order #1000

                elif holding==-1000:
                    order=2000
                    holding=holding+order #1000
                else:
                    order=0
                    holding=holding+order
            elif Y_pred[i]==-1 :
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

            trades.iloc[i]=order
        trades=pd.DataFrame(trades)	

	 		  
        return  trades
   

         	  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	   		   	 		  		  		    	 		 		   		 		  
    learner = StrategyLearner(verbose = False, impact = 0.005, commission=9.95) # constructor 
    learner.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
    trades =learner.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase
    # rmse = math.sqrt(((Y_test - Y_pred) ** 2).sum() / Y_test.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    # print()  		  	   		   	 		  		  		    	 		 		   		 		  
    # print("In sample results")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"RMSE: {rmse}")  	
    val=compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    trades,	  	   		   	 		  		  		    	 		 		   		 		  
    start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005,)
    df_bench=gen_benchmark(symbol='JPM',sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009,12,31), sv = 100000, commission=9.95, impact=0.005)
    # df_port_val=val/val.iloc[0]
    # df_bench=df_bench/df_bench.iloc[0]

    # tos.gen_plots(df_port_val,df_bench)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(val)		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_stats(df_bench)	  	   
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {val[-1]}")  
    print(f"Final Benchmark Portfolio Value: {df_bench[-1]}")