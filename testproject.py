import pandas as pd
from util import get_data, plot_data  		  	   		   	 		  		  		    	 		 		   		 		  
from marketsimcode import *
import indicators
import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt 
import experiment1 as x1
import experiment2 as x2
import random

def author(): 
  return 'sawid3'

random.seed(10)
# from  experiment1 import  *
insample=ms.testPolicy( symbol = "JPM", 
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009,12,31), 
    sv = 100000)
outsample=ms.testPolicy( symbol = "JPM", 
    sd=dt.datetime(2010, 1, 1),
    ed=dt.datetime(2011,12,31), 
    sv = 100000)
    # print(test)
manual=compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    insample,	  	   		   	 		  		  		    	 		 		   		 		  
    start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005,)
manual_out=compute_portvals(  		  	   		   	 		  		  		    	 		 		   		 		  
    outsample,	  	   		   	 		  		  		    	 		 		   		 		  
    start_val=100000,  		  	   		   	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		   	 		  		  		    	 		 		   		 		  
    impact=0.005,)
df_bench=gen_benchmark(symbol='JPM',sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009,12,31), sv = 100000, commission=9.95, impact=0.005)
df_bench2=gen_benchmark(symbol='JPM',sd=dt.datetime(2010, 1, 1),ed=dt.datetime(2011,12,31), sv = 100000, commission=9.95, impact=0.005)


manual=manual/manual.iloc[0]
manual_out=manual_out/manual_out.iloc[0]

df_bench=df_bench/df_bench.iloc[0]
df_bench2=df_bench2/df_bench2.iloc[0]
y_min = manual.min()
y_max = manual.max()
l1=[]
l2=[]
for i in insample.index:
    if insample.loc[i].item()>0:
        l1.append(i)
    if insample.loc[i].item()<0:
        l2.append(i)
fig, ax = plt.subplots()
plt.title("JPM Portfolio Values Insample Strategies")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.plot(manual, label='Manual Strategy', color='red')

plt.plot(df_bench,label='Benchmark', color='green')
plt.vlines(l1, ymin=y_min, ymax=y_max, colors='blue', label='Long Positions')
plt.vlines(l2, ymin=y_min, ymax=y_max, colors='black', label='Short Positions')

# formatter = mdates.DateFormatter("%Y-%m-%d")
fig.autofmt_xdate()  
plt.legend()
# plt.show()
plt.savefig("Insample_1.png")

y_min = manual_out.min()
y_max = manual_out.max()
l1=[]
l2=[]
for i in outsample.index:
    if outsample.loc[i].item()>0:
        l1.append(i)
    if outsample.loc[i].item()<0:
        l2.append(i)
fig, ax = plt.subplots()
plt.title("JPM Portfolio Values: Out-of-Sample Strategies")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.plot(manual_out, label='Manual Strategy', color='red')

plt.plot(df_bench2,label='Benchmark', color='green')
plt.vlines(l1, ymin=y_min, ymax=y_max, colors='blue', label='Long Positions')
plt.vlines(l2, ymin=y_min, ymax=y_max, colors='black', label='Short Positions')
# formatter = mdates.DateFormatter("%Y3o2q28q28-%m-%d")
fig.autofmt_xdate()  
plt.legend()
# plt.show()
plt.savefig("Outsample_1.png")

cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(manual)		 		   		 		  
cum_ret_b1, avg_daily_ret_b1, std_daily_ret_b1, sharpe_ratio_b1 = port_stats(df_bench)

# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Sharpe Ratio of Bench : {sharpe_ratio_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Cumulative Return of Bench : {cum_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Standard Deviation of Bench : {std_daily_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Average Daily Return of Bench : {avg_daily_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Final Portfolio Value: {manual[-1]}")  
# print(f"Final Benchmark Portfolio Value: {df_bench[-1]}")  

cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio =  	port_stats(manual_out)		 		   		 		  
cum_ret_b1, avg_daily_ret_b1, std_daily_ret_b1, sharpe_ratio_b1 = port_stats(df_bench2)
# print()
# print("OUTSAMPLE:")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Sharpe Ratio of Bench : {sharpe_ratio_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Cumulative Return of SPY : {cum_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Standard Deviation of SPY : {std_daily_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Average Daily Return of SPY : {avg_daily_ret_b1}")  		  	   		   	 		  		  		    	 		 		   		 		  
# print()  		  	   		   	 		  		  		    	 		 		   		 		  
# print(f"Final Portfolio Value: {manual_out[-1]}")  
# print(f"Final Benchmark Portfolio Value: {df_bench2[-1]}")  

s,t=x1.experiment_1()
x2.experiment_2()