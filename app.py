import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.models import load_model

def readTrain(x):
  train = pd.read_csv(x)
  return train
#2020-10-01 ===>三個class2020,10,01
def augFeatures(train):
  train = train.drop(["Date"], axis=1)
  return train  

def normalize(train):
  # train = train.drop(["Date"], axis=1)
  train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
  # print('已normalize')
  return train_norm
#第一份資料Y的norm
def denormalize(train):
  denorm = train.apply(lambda x: x*(np.max(Ytrain.iloc[:,0])-np.min(Ytrain.iloc[:,0]))+np.min(Ytrain.iloc[:,0]))
  return denorm
def add(a,b):
    add = 0.5*(a-100) + 1.5*(b-2000)
    add = add/2  
    return add
#第二份資料的norm    
def Jdenormalize(train):
  # denorm = train.apply(lambda x: x*(np.max(Ytrain.iloc[:,0])-np.min(Ytrain.iloc[:,0]))+np.min(Ytrain.iloc[:,0]))
  denorm = train.apply(lambda x: x*(np.max(Jtrain_Aug.iloc[:,0])-np.min(Jtrain_Aug.iloc[:,0]))+np.min(Jtrain_Aug.iloc[:,0]))
  # denorm = train.apply(lambda x: x*(np.max(x) - np.min(x)+np.min(x)) )
  
  # print('已denormalize')
  return denorm

#餵200天pre 60天(第一份資料)=================================================================
def buildTrainX(train, pastDay=200, futureDay=60):
   X_train = []

   for i in range(train.shape[0]-futureDay-pastDay):
    X_train.append(np.array(train.iloc[i:i+pastDay]))
    # Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["A"]))
   return np.array(X_train)

def buildTrainY(train, pastDay=200, futureDay=60):
   Y_train = []
   for i in range(train.shape[0]-futureDay-pastDay+1):
    # X_train.append(np.array(train.iloc[i:i+pastDay]))
    Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]))
   return np.array(Y_train)

# def buildTest(test):
#     x_test, y_test = [], []
#     x_test.append(np.array(test.iloc[0:150]))
#     y_test.append(np.array(test.iloc[150:157]["A"]))
#     return np.array(x_test), np.array(y_test)

def buildTestX(test):
    x_test = []
    x_test.append(np.array(test.iloc[61:261]))
    # y_test.append(np.array(test.iloc[150:157]["A"]))
    return np.array(x_test)

# def buildTestY(test):
#     y_test = []
#     # y_test.append(np.array(test.iloc[0:150]))
#     y_test.append(np.array(test.iloc[201:261]))
#     return np.array(y_test)


#餵60pre7天 第二份資料
#==================================================================================
def JbuildTrain(train, pastDay=60, futureDay=7):
     X_train, Y_train = [], []
     for i in range(train.shape[0]-futureDay-pastDay):
         X_train.append(np.array(train.iloc[i:i+pastDay]))
         Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["A"]))
     return np.array(X_train), np.array(Y_train)
def JbuildTrainX(train, pastDay=60, futureDay=7):
   X_train = []

   for i in range(train.shape[0]-futureDay-pastDay):
    X_train.append(np.array(train.iloc[i:i+pastDay]))
    # Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["A"]))
   return np.array(X_train)

def JbuildTrainY(train, pastDay=60, futureDay=7):
   Y_train = []
   for i in range(train.shape[0]-futureDay-pastDay+1):
    # X_train.append(np.array(train.iloc[i:i+pastDay]))
    Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]))
   return np.array(Y_train)

def JbuildTestX(test):
    x_test = []
    #x_test.append(np.array(test.iloc[13:73]))#3/15~3/21前60天
    x_test.append(np.array(test.iloc[20:80]))#3/23~3/30
    # y_test.append(np.array(test.iloc[150:157]["A"]))
    return np.array(x_test)

def JbuildTestY(test):
    y_test = []
    # y_test.append(np.array(test.iloc[0:150]))
    y_test.append(np.array(test.iloc[73:80]["A"]))#3/15~3/21
    return np.array(y_test)
# def DateJbuildTestY(test):
#     y_test = []
#     # y_test.append(np.array(test.iloc[0:150]))
#     y_test.append(np.array(test.iloc[67:74]))
#     return np.array(y_test)
#=======================================================================================


def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]


def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(units = 50, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(units = 50, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(units = 50, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(units = 50, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(units = 50))
    model.add(Dense(60))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

train = readTrain(x="台電2019~2020.csv")
train_Aug = augFeatures(train)
train_norm = normalize(train_Aug)
Ytrain = readTrain(x="台電2019~2020Y.csv")
# # Ytrain_Aug = augFeatures(Ytrain)
Ytrain_norm = normalize(Ytrain)
Jtrain = readTrain(x="J台電2019~2020.csv")
Jtrain_Aug = augFeatures(Jtrain)
Jtrain_norm = normalize(Jtrain_Aug)
JYtrain = readTrain(x="J台電2019~2020Y.csv")
# Ytrain_Aug = augFeatures(Ytrain)
JYtrain_norm = normalize(JYtrain)
# # change the last day and next day 
# X_train = buildTrainX(train_norm, 200,60)
# Y_train = buildTrainY(Ytrain_norm, 200,60)
# # X_train, Y_train = shuffle(X_train, Y_train)
# X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
# aaaaaaa=train_Aug[:]["A"]
# aaaaaaa=np.array(aaaaaaa)
# print(np.max(aaaaaaa))
# from 2 dimmension to 3 dimension
# Y_train = Y_train[:,:,np.newaxis]
# Y_val = Y_val[:,:,np.newaxis]

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training1',default='2021年test.csv',help='output file name')
    parser.add_argument('--training2',default='J2021年test.csv',help='output file name')
    # args = parser.parse_args()
    parser.add_argument('--output',default='submission.csv',help='output file name')
    args = parser.parse_args()
    # parser.add_argument('--training',default='training_data.csv',help='output file name')
    # args = parser.parse_args()

    datatest = pd.read_csv(args.training1)
    Jdatatest = pd.read_csv(args.training2)
    model = load_model('my_model.h5')
    Jmodel = load_model('Jmy_model.h5')
    # Ydatatest = readTrain(x='2021Y.csv')
    # Ydatatest = pd.read_csv('2021Y.csv')
    # datatest = readTrain(x='2021年test.csv')
    datatest_Aug = augFeatures(datatest)
    datatest_norm = normalize(datatest_Aug)
    Jdatatest_Aug = augFeatures(Jdatatest)
    Jdatatest_norm = normalize(Jdatatest_Aug)
#原始測試資料集
    # x_test, y_test = buildTest(datatest_Aug)
    X_test = buildTestX(datatest_norm)
    JX_test = JbuildTestX(Jdatatest_norm)

    # X_test, Y_test = buildTest(datatest_norm)
    Y_test = JbuildTestY(Jdatatest_Aug)

# Y_test = Y_test[:,:,np.newaxis]
    # Y_test = Y_test[:,:,np.newaxis]
    
    predicted_data = model.predict(X_test)
    Jpredicted_data = Jmodel.predict(JX_test)
    predicted_data = pd.DataFrame(np.concatenate(predicted_data))
    Jpredicted_data = pd.DataFrame(np.concatenate(Jpredicted_data))
    predicted_data = denormalize(predicted_data)
    Jpredicted_data = Jdenormalize(Jpredicted_data)
    # y_hat = predicted_data.iloc[37:44]#3/9號~3/15
    # Jy_hat = Jpredicted_data.iloc[0:7]#3/9號~3/15
    # y_hat = predicted_data.iloc[43:50]#3/15~3/21
    y_hat = predicted_data.iloc[51:58]#3/23~3/30
    Jy_hat = Jpredicted_data.iloc[0:7]
    y_hat = np.array(y_hat)
    Jy_hat = np.array(Jy_hat)
    #兩個model參數調整
    final_pre = add(y_hat,Jy_hat)

    # print(rmse(final_pre,Y_test))

    
    final_pre = pd.DataFrame(final_pre)
    # Y_test = pd.DataFrame(Y_test)
    y_test = JbuildTestY(Jdatatest)
    y_test = np.reshape(y_test,(7,1))
    y_test = pd.DataFrame(y_test)
    # model.train(df_training)
    # df_result = model.predict(n_step=7)
    # y_hat1 = DataFrame(y_hat,index = ['20210323','20210324','20210325','20210326','20210327','20210328','20210329'],columns=['0'])
    final_pre.index = Series(['03-23','03-24','03-25','03-26','03-27','03-28','03-29'])
    y_test.index = Series(['03-23','03-24','03-25','03-26','03-27','03-28','03-29'])
    # final_pre.index = Series(['03-09','03-10','03-11','03-12','03-13','03-14','03-15'])
    # y_test.index = Series(['03-09','03-10','03-11','03-12','03-13','03-14','03-15'])
    #畫圖
    plt.xlabel('Date', fontsize = 16)                        # 設定坐標軸標籤
    plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
    plt.yticks(fontsize = 12)
    # plt.grid(color = 'red', linestyle = '--', linewidth = 1)  # 設定格線顏色、種類、寬度
    plt.ylim(2000, 4000)                                          # 設定y軸繪圖範圍
# 繪圖並設定線條顏色、寬度、圖例
    line1, = plt.plot(final_pre, color = 'red', linewidth = 3, label = 'predict')             
    line2, = plt.plot(y_test, color = 'blue', linewidth = 3, label = 'ground true')
    plt.legend(handles = [line1, line2])
    # plt.savefig('Fe_r_plot.svg')                              # 儲存圖片
    # plt.savefig('Fe_r_plot.png')
    plt.show()   

    final_pre.to_csv(args.output)