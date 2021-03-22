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

def denormalize(train):
  denorm = train.apply(lambda x: x*(np.max(Ytrain.iloc[:,0])-np.min(Ytrain.iloc[:,0]))+np.min(Ytrain.iloc[:,0]))
  # denorm = train.apply(lambda x: x*(np.max(train_Aug.iloc[:,0])-np.min(train_Aug.iloc[:,0]))+np.min(train_Aug.iloc[:,0]))
  # denorm = train.apply(lambda x: x*(np.max(x) - np.min(x)+np.min(x)) )
  
  # print('已denormalize')
  return denorm

#30天做為過去資料 30天作為欲預測資料
# def buildTrain(train, pastDay=150, futureDay=7):
#    X_train, Y_train = [], []

#    for i in range(train.shape[0]-futureDay-pastDay):
#     X_train.append(np.array(train.iloc[i:i+pastDay]))
#     Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["A"]))
#    return np.array(X_train), np.array(Y_train)
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
    x_test.append(np.array(test.iloc[1:201]))
    # y_test.append(np.array(test.iloc[150:157]["A"]))
    return np.array(x_test)

def buildTestY(test):
    y_test = []
    # y_test.append(np.array(test.iloc[0:150]))
    y_test.append(np.array(test.iloc[226:260]))
    return np.array(y_test)

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
# Ytrain_Aug = augFeatures(Ytrain)
Ytrain_norm = normalize(Ytrain)
# change the last day and next day 
X_train = buildTrainX(train_norm, 200,60)
Y_train = buildTrainY(Ytrain_norm, 200,60)
# X_train, Y_train = shuffle(X_train, Y_train)
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
# aaaaaaa=train_Aug[:]["A"]
# aaaaaaa=np.array(aaaaaaa)
# print(np.max(aaaaaaa))
# from 2 dimmension to 3 dimension
# Y_train = Y_train[:,:,np.newaxis]
# Y_val = Y_val[:,:,np.newaxis]

## 訓練LSTM
# model = buildManyToManyModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), callbacks=[callback])
# model.save('my_model.h5')



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',default='2021年test.csv',help='output file name')
    # args = parser.parse_args()
    parser.add_argument('--output',default='submission.csv',help='output file name')
    args = parser.parse_args()
    # parser.add_argument('--training',default='training_data.csv',help='output file name')
    # args = parser.parse_args()

    datatest = pd.read_csv(args.training)
    model = load_model('my_model.h5')

    # Ydatatest = readTrain(x='2021Y.csv')
    Ydatatest = pd.read_csv('2021Y.csv')
    # datatest = readTrain(x='2021年test.csv')
    datatest_Aug = augFeatures(datatest)
    datatest_norm = normalize(datatest_Aug)
#原始測試資料集
    # x_test, y_test = buildTest(datatest_Aug)
    X_test = buildTestX(datatest_norm)
#規一化資料集
    # X_test, Y_test = buildTest(datatest_norm)
    Ydatatest_Aug = augFeatures(Ydatatest)
    Y_test = buildTestY(Ydatatest_Aug)

# Y_test = Y_test[:,:,np.newaxis]
    # Y_test = Y_test[:,:,np.newaxis]
    
    predicted_data = model.predict(X_test)
    predicted_data = pd.DataFrame(np.concatenate(predicted_data))
    predicted_data1 = denormalize(predicted_data)
    y_hat = predicted_data1.iloc[26:60].values
    y_hat = pd.DataFrame(y_hat)
    y_hat1 = np.array(y_hat)[np.newaxis,:,:]
        #畫圖
    Y_test = np.reshape(Y_test,(34,1))    
    plt.xlabel('2021/01/01~2021/02/06', fontsize = 16)                        # 設定坐標軸標籤
    plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
    plt.yticks(fontsize = 12)
    # plt.grid(color = 'red', linestyle = '--', linewidth = 1)  # 設定格線顏色、種類、寬度
    plt.ylim(2000, 4000)                                          # 設定y軸繪圖範圍
# 繪圖並設定線條顏色、寬度、圖例
    line1, = plt.plot(y_hat, color = 'red', linewidth = 3, label = 'predict')             
    line2, = plt.plot(Y_test, color = 'blue', linewidth = 3, label = 'ground true')
    plt.legend(handles = [line1, line2])
    # plt.savefig('Fe_r_plot.svg')                              # 儲存圖片
    # plt.savefig('Fe_r_plot.png')
    plt.show()  
    # Y_test = pd.DataFrame(np.concatenate(Y_test))
    # realdata = denormalize(Y_test)
    # print(rmse(y_hat1,Y_test))
    # model.train(df_training)
    # df_result = model.predict(n_step=7)
    # y_hat1 = DataFrame(y_hat,index = ['20210323','20210324','20210325','20210326','20210327','20210328','20210329'],columns=['0'])
    # y_hat.index = Series(['2021-03-23','2021-03-24','2021-03-25','2021-03-26','2021-03-27','2021-03-28','2021-03-29'])
    # a
    y_hat.to_csv(args.output)
