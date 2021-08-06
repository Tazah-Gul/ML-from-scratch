import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class SimpleLinearRegression:
    def __init__(self,x,y,learning_rate,epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.m = x.shape[0]
        self.x = x
        self.y = y
        self.weight = -1
        self.bias = 0

    def fit(self):
        for i in range(1,self.epochs+1):
            z = self.hypothesis()
            j = self.cost_function(z)
            if i%50==0:
                print("cost",j)
            dw,db = self.gradient(z)
            self.gradient_descent_update(dw,db)

    def predict(self,x):
        result = self.weight*x+self.bias
        return result

    def hypothesis(self):
        z = self.weight*self.x+self.bias
        return z

    def cost_function(self,z):
        j = (0.5/self.m) * np.sum(np.square(z-self.y))
        return j

    def gradient(self,z):
        dz = (1/self.m)*(z-self.y)
        db = np.sum(dz)
        dw = np.sum(dz*self.x)
        return dw,db

    def gradient_descent_update(self, dw, db):
        self.weight = self.weight - self.lr*dw
        self.bias = self.bias - self.lr*db

    def graph(self, param):
        x = np.array(param)
        y = self.predict(x)
        plt.plot(x,y)


if __name__ == '__main__':
    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    x_train, y_train = data_train["x"], data_train["y"]
    x_test, y_test = data_test["x"], data_test["y"]
    x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # Lets make an object of class to train our model on train data
    train_model = SimpleLinearRegression(x_train,y_train,0.1,1000)
    train_model.fit()
    # Lets make an object of class to test our model on test data
    print("Now lets test our model using test data")
    x_test = (x_test - x_test.mean()) / (x_test.max() - x_test.min())
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    test_model = SimpleLinearRegression(x_test, y_test, 0.1, 1000)
    test_model.fit()
    #Lets compare our result with Linear Regression model from sklearn
    regressor = LinearRegression()
    regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))
    y_pred = regressor.predict(x_test.reshape(-1,1))
    print(r2_score(y_test,y_pred))
    # Lets make a graph of train data and draw best fit line on it
    plt.scatter(x_train, y_train, c="red", alpha=.5, marker="o")
    train_model.graph(range(-1,2))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
