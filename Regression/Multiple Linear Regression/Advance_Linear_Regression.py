import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def initialize_parameters(lenw):    #step1
    w = np.random.randn(lenw)
    b = 0
    return w,b


def hypothesis(x,w,b):  #step2
    z = np.dot(w,x) + b   # w--->1*n  x--->n*m   b is broadcasted into vector  [b b b.......b] since its originally scalar
    return z

def cost_function(z,y):   #step3
    m = y.shape[1]
    j = (0.5/m) *np.sum(np.square(z-y))
    return j
def Grad(x,y,z):     #step4
    m = y.shape[1]
    dz = (1/m) * (z-y)
    dw = np.dot(dz,x.T)
    db = np.sum(dz)
    return dw,db
def grad_descent_update(w,b,dw,db,lr):      #step5
    w = w - lr * dw
    b = b - lr * db
    return w,b


def linear_regression_model(x_train,y_train,x_val,y_val,lr,epochs):     #step6

    lenw = x_train.shape[0]
    w,b = initialize_parameters(lenw)   #step1

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1,epochs+1):
        z_train = hypothesis(x_train,w,b)         #step2
        cost_train = cost_function(z_train,y_train)        #step3
        dw,db = Grad(x_train,y_train,z_train)     #step4
        w,b = grad_descent_update(w,b,dw,db,lr)    #step5
        #storing training cost for plotting purpose
        if i%10==0:
            costs_train.append(cost_train)
        #MAE Train
        MAE_train = (1/m_train) * np.sum(np.abs(z_train-y_train))

        #cost_val, MAE value
        z_val = hypothesis(x_val, w, b)
        cost_val = cost_function(z_val, y_val)
        MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_val))

        #print out cost_train, cost_val,MAE_train,MAE_val
        print("Epochs "+str(i)+"/"+str(epochs)+":")
        print("Training cost " + str(cost_train) + "|"+"Validation cost " + str(cost_val))
        print("MAE cost " + str(MAE_train) + "|" + "Validation MAE " + str(MAE_val))
    plt.plot(costs_train)
    plt.xlabel('Iterations(per ten)')
    plt.ylabel('Training cost')
    plt.title("Learning rate" + str(lr))
    plt.show()
    return z_val

if __name__ == '__main__':
    data = pd.read_csv("housing.csv")
    x = data[["RM", "LSTAT", "PTRATIO"]]
    x = (x - x.mean())/(x.max() - x.min())
    y = data["MEDV"]
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.33,random_state=5)
    x_train = x_train.T
    y_train = np.array([y_train])
    x_val = x_val.T
    # x_val2 = [[6.03 ,7.88 ,21.0]]
    y_val = np.array([y_val])
    lr = 1.9993
    x_val2 = x_val.T[2:3]
    print(x_train.shape)
    # print(x_val[0])
    # z_val = linear_regression_model(x_train,y_train,x_val2.T,y_val,lr,5000)
    # print(z_val)
    # print(x_val.T[2:3])
    #Lets compare our result with Linear Regression model from sklearn
    # linear_regression = linear_model.LinearRegression()
    # model = linear_regression.fit(x_train.T,y_train.T)
    # predictions = linear_regression.predict(x_val.T[2:3])
    # MAE_value_with_sklearn = (1/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))
    # print(MAE_value_with_sklearn)
    # print(predictions)
    # data.info()
    # data.describe()




