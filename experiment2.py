import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
from marketsimcode import *
import indicators
import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt 

def author(): 
  return 'sawid3'
# hypothesis average daily return and sharp ratio goes down as impact increases
def experiment_2():
# impact 0.01
    learner1 = sl.StrategyLearner(verbose = False, impact = 0.01, commission=0) # constructor 
    learner1.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
    strategy_trade1 =learner1.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase

    strategy1=compute_portvals(	   		   	 		  		  		    	 		 		   		 		  
        strategy_trade1,	  	   		   	 		  		  		    	 		 		   		 		  
        start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
        commission=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        impact=0.01,)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(strategy1)		 		   		 		  
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_stats(df_bench)	  	   
    print() 
    # print(f"Sharpe Ratio of Ex 1: {sharpe_ratio}")  	
    # print(f"Cumulative Return of  Ex 1: {cum_ret}")  
    # print(f"Standard Deviation of Ex 1: {std_daily_ret}")  		
    # print(f"Average Daily Return of Ex 1: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  



    # impact 0.05
    learner2 = sl.StrategyLearner(verbose = False, impact = 0.05, commission=0) # constructor 
    learner2.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
    strategy_trade2 =learner2.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase
    strategy2=compute_portvals(	   		   	 		  		  		    	 		 		   		 		  
        strategy_trade2,	  	   		   	 		  		  		    	 		 		   		 		  
        start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
        commission=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        impact=0.05,)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(strategy2)		 		   		 		  

    # print() 
    # print(f"Sharpe Ratio of Ex 2: {sharpe_ratio}")  	
    # print(f"Cumulative Return of  Ex 2: {cum_ret}")  
    # print(f"Standard Deviation of Ex 2: {std_daily_ret}")  		
    # print(f"Average Daily Return of Ex 2: {avg_daily_ret}")

    # impact 0.0001
    learner3 = sl.StrategyLearner(verbose = False, impact = 0.0001, commission=0) # constructor 
    learner3.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
    strategy_trade3 =learner3.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase
    strategy3=compute_portvals(	   		   	 		  		  		    	 		 		   		 		  
        strategy_trade3,	  	   		   	 		  		  		    	 		 		   		 		  
        start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
        commission=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        impact=0.0001,)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(strategy3)		 		   		 		  
    print() 
    # print(f"Sharpe Ratio of Ex 3: {sharpe_ratio}")  	
    # print(f"Cumulative Return of  Ex 3: {cum_ret}")  
    # print(f"Standard Deviation of Ex 3: {std_daily_ret}")  		
    # print(f"Average Daily Return of Ex 3: {avg_daily_ret}")

    fig, ax = plt.subplots()
    plt.title("JPM Portfolio Values Across Different Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.plot(strategy2, label='impact 0.05', color='red')
    plt.plot(strategy1, label='impact 0.01', color='black')
    plt.plot(strategy3, label='impoact 0.0001', color='green')

    # plt.plot(df_bench,label='Benchmark', color='black')
    # formatter = mdates.DateFormatter("%Y-%m-%d")
    fig.autofmt_xdate()  
    plt.legend()
    # plt.show()
    plt.savefig("Expirment2.png")