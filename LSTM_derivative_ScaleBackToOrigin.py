import os
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

# set hyperparameters
timestep = 1
batchsize = 75
epochs = 200
n_features = 9


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


filename = ['1.csv', '2.csv', '3.csv']
trainfoldpath = 'D:/Studium/Masterarbeit/Datenvorbearbeiten/train/'
valifoldpath = 'D:/Studium/Masterarbeit/Datenvorbearbeiten/validation/'
modelname = 'lstm_model.h5'
# load dataset
for i in range(len(filename)):
    traindataset = read_csv(trainfoldpath + filename[i], usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    validataset = read_csv(valifoldpath + filename[i], usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trainvalues = traindataset.values[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
    valivalues = validataset.values[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
    # ensure all data is float
    trainvalues = trainvalues.astype('float32')
    valivalues = valivalues.astype('float32')

    # normalize features
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_vali = MinMaxScaler(feature_range=(0, 1))
    scaledtrain = scaler_train.fit_transform(trainvalues)
    scaledvali = scaler_vali.fit_transform(valivalues)
    # frame as supervised learning
    reframedtrain = series_to_supervised(scaledtrain, 1, 1)
    reframedvali = series_to_supervised(scaledvali, 1, 1)

    # split into input and outputs
    train = reframedtrain.values
    vali = reframedvali.values
    train_X, train_y = train[:, :-1], train[:, -1]
    vali_X, vali_y = vali[:, :-1], vali[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    vali_X = vali_X.reshape((vali_X.shape[0], 1, vali_X.shape[1]))

    # design network
    if not os.path.exists(modelname):
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        print('a new model has been created')
    else:
        model = load_model(modelname)
        print('the old model has been loaded')
    # fit network
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batchsize, validation_data=(vali_X, vali_y), verbose=2, shuffle=False)
    model.save(modelname)
    print('File {} has been trained, model {} has been saved'.format(filename[i], modelname))
'''
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.close()
'''
# prepare testdata
testfile = '1'
testdataset = read_csv('D:/Studium/Masterarbeit/Datenvorbearbeiten/test/{}.csv'.format(testfile), usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
testvalues = testdataset.values[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]]
testvalues = testvalues.astype('float32')
max_testy = max(testvalues[:, -1])
min_testy = min(testvalues[:, -1])
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaledtest = scaler_test.fit_transform(testvalues)
reframedtest = series_to_supervised(scaledtest, 1, 1)
test = reframedtest.values
test_X, test_y = test[:, :-1], test[:, -1]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# make a prediction
model = load_model(modelname)
yhat = model.predict(test_X)

# check before inv_scale
plt.plot(test_y, label='label')
plt.plot(yhat, label='predict')
plt.legend()
plt.show()
plt.close()

# inv_scale
lenth_y = test_y.shape[0]
test_y = test_y.reshape(test_y.shape[0], 1)
scaler_inv = MinMaxScaler(feature_range=(min_testy, max_testy))
yall = concatenate((yhat, test_y), axis=0)
inv_yall = scaler_inv.fit_transform(yall)
print(inv_yall[:lenth_y, 0])
inv_yhat = inv_yall[:lenth_y, 0]
inv_y = inv_yall[lenth_y:, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# plot abteilung prediction
# plt.plot(testvalues[:, -1], label='label')
plt.plot(inv_y, label='label')
plt.plot(inv_yhat, label='predict')
plt.legend()
plt.show()
plt.close()
# plot
zhat = [testdataset.values[0, 9], ]
for i in range(1, len(inv_y)):
    zhat.append(zhat[-1] + inv_yhat[i])
zhat = np.array(zhat)
z = testdataset.values[:-1, 9]
rmse_z = sqrt(mean_squared_error(z, zhat))
print('Test Z RMSE: %.3f' % rmse_z)
plt.plot(z, label='label')
plt.plot(zhat, label='predict')
plt.legend()
plt.show()

# delete model
os.remove(modelname)
