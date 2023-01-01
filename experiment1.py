import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
from marketsimcode import *
import indicators
import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt 

def author(): 
  return 'sawid3'

def experiment_1():
    manual_trade=ms.testPolicy( symbol = "JPM", 
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009,12,31), 
        sv = 100000)

    learner = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95) # constructor 
    learner.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase 
    strategy_trade =learner.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # testing phase

    strategy=compute_portvals(	   		   	 		  		  		    	 		 		   		 		  
        strategy_trade,	  	   		   	 		  		  		    	 		 		   		 		  
        start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
        commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
        impact=0.005,)

    manual=compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
        manual_trade,	  	   		   	 		  		  		    	 		 		   		 		  
        start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
        commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
        impact=0.005,)

    df_bench=gen_benchmark(
        symbol='JPM',sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009,12,31), sv = 100000, 
        commission=9.95, impact=0.005)


    strategy=strategy/strategy.iloc[0]
    manual=manual/manual.iloc[0]

    df_bench=df_bench/df_bench.iloc[0]
    # def gen_plots(tos_df, df_bench ):
    fig, ax = plt.subplots()
    plt.title("JPM Portfolio Values Across Different Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.plot(strategy, label='Strategy Learner', color='red')
    plt.plot(manual, label='Manual Strategy', color='blue')

    plt.plot(df_bench,label='Benchmark', color='black')
    # formatter = mdates.DateFormatter("%Y-%m-%d")
    fig.autofmt_xdate()  
    plt.legend()
    # plt.show()
    plt.savefig("Expirment1.png")
    return strategy, manual
    
if __name__ == "__main__":
    experiment_1()