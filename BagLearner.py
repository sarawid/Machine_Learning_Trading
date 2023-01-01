import numpy as np

import RTLearner as dt 	
from scipy import stats	  	   
# import random
# from random import choice	  

def author(): 
  return 'sawid3' 	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is  using the same learning algorithm but train each learner on a different set of the data Learner. 	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self,learner=dt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  
        self.models=[learner(**kwargs) for i in range(bags)] # store model in a list of m bags
        self.bags=bags		  	   		   	 		  		  		    	 		 		   		 		  
        self.boost = boost
        self.verbose = verbose 
        self.learner=learner

    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "sawid3"  # replace tb34 with your Georgia Tech username  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """  
        if self.boost:
            pass
        rand_range=data_x.shape[0] #get the number of data available
        for model in self.models:
            # code source :https://stackoverflow.com/questions/43281886/get-a-random-sample-with-replacement/43281974

            rand_i=np.random.choice(rand_range, rand_range)
            model.add_evidence(data_x[rand_i], data_y[rand_i])	




    def query(self, points):

        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  
        """	   		   	 		  	
        res = []
        for i in range(self.bags):
            res.append(self.models[i].query(points))
            if self.verbose:
                print (res[i])
        # vals,counts = np.unique(res, return_counts=True)
        # index = np.argmax(counts)
        # pred=vals[index]
        pred = np.asarray(res)
        pred = stats.mode(pred, axis=0)
        pred = pred[0]
        # pred=np.mean(res, axis = 0)
        return pred