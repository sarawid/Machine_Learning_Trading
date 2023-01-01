import numpy as np 
# import random  		  	   		   	 		  		  		    	 		 		   		 		  

def author(): 
  return 'sawid3'
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class RTLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Decision Tree Regression Learner. 	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  
        Note code logic implemented as per class notes from decision tree video 2 based on A Cutler	  		    	 		 		   		 		  
 	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        self.verbose = verbose 
         # move along, these aren't the drones you're looking for  		  	   		   	 		  		  		    	 		 		   		 		  

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
        self.model = self.build_tree(data_x, data_y)


    def  best_feature(self,data_x, data_y ) :
        """
        returns best factor to split on based on which factor has the highest correlation with the y
        """  	
        rand_int=np.random.randint(data_x.shape[1])

        return rand_int

    def build_tree(self, data_x, data_y):
        """

        checks for stopping crietria 
        returns an array with root, left tree, right tree or leaf nodes
        note: the logic is adpated from decision tree video 1 
        """	   	 		  		 		    	 
        if self.verbose:
            print( "data", data_x)
            print( "data.shape", data_x.shape)
            # print()
        		 		   		 		  
        if data_x.shape[0]==1:
            return np.array([[-1, data_y, np.nan, np.nan]]) # if only one row
        if np.all(data_y[:] == data_y[0], axis = 0): # if all values are the same
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        if data_x.shape[0]<= self.leaf_size: # if number of rows is less than leaf size
            vals,counts = np.unique(data_y, return_counts=True)
            index = np.argmax(counts)
            return np.array([[-1, vals[index], np.nan, np.nan]])

        else:
            i=self.best_feature(data_x, data_y )
            split= np.median(data_x[:,i])


            b=max(data_x[:,i])
            if b==split:
                # array = np.array([1,2,2,4,4,5])
                vals,counts = np.unique(data_y, return_counts=True)
                index = np.argmax(counts)
                return np.array([[-1, vals[index], np.nan, np.nan]])



        if self.verbose:
            print("best feature", i)
            print("median", np.median(data_x[:,i]))
            print("left", data_x[data_x[:,i]<=split])
            print("right", data_x[data_x[:,i]>split])

        left=self.build_tree(data_x[data_x[:,i]<=split],data_y[data_x[:, i] <= split] )
        right=self.build_tree(data_x[data_x[:,i]>split],data_y[data_x[:,i]>split] )
        root = np.array([[i, split, 1, left.shape[0] + 1]])
        return np.concatenate((root, left, right), axis=0)

    def query(self, points):

        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  	
        note code logic is adapted from :
         https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea	  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        pred=np.zeros(points.shape[0])
        i=0
        for point in points :
            if self.verbose:
                print("------------------------------", point)
            T = 0
            factor = int(self.model[T,0]) # get the first feature to split on 
            while (factor !=-1): # check to split on factors only 
                split = self.model[T,1] # find the split val
                left = int(self.model[T,2]) # get left tree
                right = int(self.model[T,3]) # get right tree
                if(point[factor] <= split): # if current value is less than split go to left
                    T = T + left
                else:
                    T = T + right
                factor = int(self.model[T,0]) #once row is updatet get the new feature that is T rows down from current tree
            pred[i]+=self.model[T,1] # append terminal value
            i+=1 # append terminal value
        # pred=np.array(pred )
        # pred=pred.ravel()
        return pred
      		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		   	 		  		  		    	 		 		   		 		  
