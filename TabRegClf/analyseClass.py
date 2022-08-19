"""
    Generic tabular data modeling:
    This module aims to build Regression or Classification models for any type 
    of tabular data, through the use of autoML.

    @author Pedro G. Dubiela
"""

import pandas    as pd
import numpy     as np
import autokeras as ak
import copy
from   sklearn.model_selection import train_test_split
from   sklearn.metrics         import accuracy_score, f1_score, mean_squared_error

class TabularAnalyse:
    """
    A class to represent a TabularAnalyse.
    ...
      
    Attributes
    ----------
    __data : pd.core.frame.DataFrame
        The database.
    __target : str
        The name of the target variable.
    __analyse_type : str
        If is Regression or Classification.
    __best_model : str
        The best model obtained.
    __summary: str
        The summary of best model obtained..
    __yp : str
        Predsiction values..
    __performance : str
       The performance of best model obtained.
 
    """
    
    def __init__(self, data, target, analyse_type):
        """
        This method constructs all the necessary attributes for the 
        TabularAnalyses object

        Parameters
        ----------
        data : pd.core.frame.DataFrame
            The database.
        target : str
            The name of the target variable.
        analyse_type : str
            If is Regression or Classification.
    
        """
        # Validate
        self.__input_check(data, target, analyse_type)
        self.__data = data
        self.__target = target
        self.__analyse_type = analyse_type
        
    def __input_check(self, data, target, analyse_type):
        """
        This method checks if the inputs are correctly.

        Parameters
        ----------
        data : pd.core.frame.DataFrame
            The database.
        target : str
            The name of the target variable.
        analyse_type : str
            If is Regression or Classification.

        Raises
        ------
        Exception
            It used for checks the inputs.
        """
        if not isinstance(data, pd.core.frame.DataFrame):
            raise Exception(">>> data must be a Pandas's DataFrame")
        
        if (not isinstance(target, str)) or (target not in data.columns):
            raise Exception(
                ">>> target must be a string with the target variable name"
                )
            
        if not isinstance(analyse_type, str) or analyse_type not in (
                "Regression", "Classification"
                ) :
            raise Exception(
                ">>> analyse_type must be  Regression ou Classification"
                )
       
    #################################Getter Methods############################  
    def get_data(self):
        return self.__data
    
    def get_target(self):
        return self.__target

    def get_analyse_type(self):
        return self.__analyse_type
    
    def get_prediction(self):
        return self.__yp
    
    def get_performance(self):
        return self.__performance
    ###########################################################################
    
    def create_model(self, test_size = .2, max_trials = 5, random_state = 25, **kwargs):
        """
        This method creates, trains, tests and evaluates the model.
        
        Parameters
        ----------
        test_size : TYPE, optional
            DESCRIPTION. The default is .2.
        max_trials : TYPE, optional
            DESCRIPTION. The default is 5.
        """
        
        # split into train and test 
        df = copy.deepcopy(self.__data)
        y = df.pop(self.__target)
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
        
        switcher = {
            "Classification": ak.StructuredDataClassifier,
            "Regression": ak.StructuredDataRegressor
        }
        
        model = switcher.get(self.__analyse_type)(overwrite = True, 
                                                  max_trials = max_trials)
        
        # train
        print("Performs training")
        model.fit(x = X_train, y = y_train, **kwargs)
        
        # summary
        self.__best_model = model.export_model()
        self.__summary = self.__best_model.summary()
        print(self.__summary)
        
        
        # prediction
        print("Performs perdiction")
        self.__yp = model.predict(X_test)
        
        if (self.__analyse_type == "Regression"):
            self.__performance = mean_squared_error(y_test, self.__yp)
        else:
            self.__performance = accuracy_score(y_test, self.__yp)
            
        print(self.__performance)
        print("Performed successfully")
