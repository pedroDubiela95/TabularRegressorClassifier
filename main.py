import pandas                  as     pd
import numpy                   as     np
import TabRegClf.analyseClass  as     cl
import autokeras               as     ak
from   tensorflow.keras.models import load_model
import tensorflow              as     tf
import matplotlib.pyplot       as plt

if(__name__ == '__main__'):
            
    # Creating any dataset that has a pattern
    n = 1000
    x1 = np.linspace(0, 100, n)
    x2 = np.linspace(25, 75, n)
    x3 = np.linspace(40, 60, n)
    z = x1 + 2*x2**2 - x3
    df = pd.DataFrame({"var1": x1, "var2": x2, "var3": x3, "var4": z})
    df.to_csv("./data/data.csv")
    
    #Load data
    data = pd.read_csv("./data/data.csv", index_col=[0])
    target = 'var4'
    analyse_type = 'Regression'
    
    # Create, train and test
    tab = cl.TabularAnalyse(data, target, analyse_type)
    tab.get_analyse_type()
    tab.get_data()
    tab.get_target()
    tab.create_model(max_trials= 3, epochs = 10000)
    
    # Load best model obtained
    loaded_model = load_model("./structured_data_regressor/best_model", 
                              custom_objects=ak.CUSTOM_OBJECTS)
    
    # Let's apply the best model on all database
    x_test = data.iloc[[0], :-1] 
    x_test = data.iloc[:, :-1] 
    yp = loaded_model.predict(x_test)
    yt = data.iloc[:, -1].values.reshape(-1, 1)
    error_per = np.abs(yp - yt)/yt * 100
    plt.hist(error_per)
    plt.title("Performance of Best Model")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Frequency")
    plt.savefig("./images/fig")
 
