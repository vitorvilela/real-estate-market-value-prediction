import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm as cm

import pandas as pd
from pandas.io.json import json_normalize
from pandas import read_csv
from pandas import set_option

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras import optimizers
from keras import metrics

EPS = 1e-12

font_normal = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontsize'   : 16 }

font_italic = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontstyle'  : 'italic',
                'fontsize'   : 16 }

font_italic_labels = { 'color'      : 'k',
                       'fontweight' : 'normal',
                       'fontstyle'  : 'italic',
                       'fontsize'   : 12 }


markers = ('k-', 'k:')
  
# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# O coeficiente de correlação de Pearson, rxy ∈ [−1, +1], fornece uma medida da relação linear entre duas variáveis
# O coeficiente de correlação de Spearman, rs ∈ [−1, +1], por sua vez, indica se duas variáveis são monotônicas, independentemente da relação linear. Adicionalmente, o coeficiente de Spearman é menos sensível aos outliers de uma amostra do que o coeficiente de Pearson.

#pearson : standard correlation coefficient
#spearman : Spearman rank correlation
#kendall : Kendall Tau correlation coefficient
#callable : callable with input two 1d ndarrays
corr_method = 'spearman'

# File name options
filename = './dataset/source-4-ds-train.json'
#filename = './dataset/train_listOfDict.json'
#filename = './dataset/train_dictPerLine.json'
#filename = './dataset/train_example.json'

# Open a JSON file as a list of dicts template and return a list of dicts
#with open(filename, 'r') as f:  
 #dict_list = json.load(f)
 
# Open a JSON file as a dict per line template and return a pandas dataFrame
# dtype={'address.neighborhood': str, 'unitTypes': str})
df_origin = pd.read_json(filename, lines=True) 

# Convert a pandas dataFrame into a list of dicts
dict_list = df_origin.to_dict('records')

# Expand nested JSON into a pandas dataFrame
dataframe_original = json_normalize(dict_list) 
# 133.964 elements

# Cleaning / Restricting database

# Just properties for sale
dataframe_sale = dataframe_original.loc[dataframe_original['pricingInfos.businessType'] == 'SALE']
# 105.332 elements

# Just APARTMENT
dataframe_apartment = dataframe_sale.loc[dataframe_sale['unitTypes'] == 'APARTMENT']
# 64.146 elements

# Cleaning disparate data 
dataframe_clean = dataframe_apartment.loc[dataframe_apartment['pricingInfos.price'] >= 3.e4] # >= 1.e5]
# 64.052 (leaving 94 low price apartments outside - look later their description and features)
dataframe_clean = dataframe_clean.loc[dataframe_clean['pricingInfos.price'] < 4.e7] # < 1.e7]
# 63.933 (leaving 119 high price apartments outside - look later their description and features)

# Grouping data (look later their description and features)
dataframe_group = dataframe_clean.loc[dataframe_clean['pricingInfos.price'] >= 1.e6] # < 1.e6]
# below million 51.120
# above million 12.813

dataframe = dataframe_group.loc[:,['pricingInfos.price', 'bathrooms', 'bedrooms', 'parkingSpaces', 'suites', 'usableAreas']]
dataframe = dataframe.loc[:, :].apply(pd.to_numeric)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print([c for c in dataframe.columns])
  print(dataframe.head())

#address.city                         0                                                                                                                                                                                                                                                                                      
#address.country                      1                                                                                                                                                                                                                                                                                      
#address.district                     2                                                                                                                                                                                                                                                                                      
#address.geoLocation.location.lat    3                                                                                                                                                                                                                                                                                      
#address.geoLocation.location.lon    4                                                                                                                                                                                                                                                                                      
#address.geoLocation.precision        5                                                                                                                                                                                                                                                                                      
#address.locationId                   6                                                                                                                                                                                                                                                                                      
#address.neighborhood                 7                                                                                                                                                                                                                                                                                      
#address.state                        8                                                                                                                                                                                                                                                                                      
#address.street                       9                                                                                                                                                                                                                                                                                      
#address.streetNumber                 10                                                                                                                                                                                                                                                                                      
#address.unitNumber                   11                                                                                                                                                                                                                                                                                    
#address.zipCode                      12                                                                                                                                                                                                                                                                                      
#address.zone                         13                                                                                                                                                                                                                                                                                      
#bathrooms                           14        *                                                                                                                                                                                                                                                                              
#bedrooms                            15        *                                                                                                                                                                                                                                                                             
#createdAt                            16                                                                                                                                                                                                                                                                                      
#description                          17                                                                                                                                                                                                                                                                                      
#id                                   18                                                                                                                                                                                                                                                                                      
#images                               19                                                                                                                                                                                                                                                                                      
#listingStatus                        20                                                                                                                                                                                                                                                                                      
#owner                                  21                                                                                                                                                                                                                                                                                      
#parkingSpaces                       22        *
#pricingInfos.businessType            23
#pricingInfos.monthlyCondoFee        24
#pricingInfos.period                  25
#pricingInfos.price                    26       **
#pricingInfos.rentalTotalPrice       27
#pricingInfos.yearlyIptu             28
#publicationType                      29
#publisherId                          30
#suites                              31       *
#title                                32
#totalAreas                          33
#unitTypes                            34
#updatedAt                            35
#usableAreas                         36       *

dataset = dataframe.values
print('original ', dataset.shape)

dataset = dataset[~np.isnan(dataset).any(axis=1), :]
print('without nan ', dataset.shape)

dataset = dataset[~np.isinf(dataset).any(axis=1), :]
print('without inf ', dataset.shape)

X = dataset[:,1:]  # dataset[:,np.r_[14,15,22,31,36]] 
Y = dataset[:,0]
print(X.shape)
print(Y.shape)

test_size = 0.2 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Scaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Model parameters
batch_size = int(X_train.shape[0]/1)

epochs = 15000

kernel_initializer = 'normal' # 'uniform'

loss = 'msle' # 'mse' 'mape' 'mae' 'msle' 'logcosh'

activation = 'selu'
actname = 'selu'
# 'relu' 'sigmoid'
#keras.activations.elu(x, alpha=1.0)
#keras.activations.selu(x)
#keras.layers.LeakyReLU(alpha=0.3)
#keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

metrics = ['mse', 'mae', 'mape']  # 'mse' 'mae', 'mape', 'cosine' for regression; 'acc' for classification;

layers = '2'
neurons = '10x'
batchnorm = 'yes'
drop = 'yes'

# Define the model
model = Sequential()

# model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer, activation=activation))
model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))

model.add(Dense(10*X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))

model.add(Dense(10*X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))

#model.add(Dense(Y_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dense(1, kernel_initializer=kernel_initializer))

learning_rate = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]
lr = 1.e-1

lrdecay = 1.e-4

loss_list = []

adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lrdecay, amsgrad=False)
sgd = optimizers.SGD(lr=lr, decay=lrdecay, momentum=0.9, nesterov=True)
rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=lrdecay)

optimizer = adam
optname = 'adam'
# 'adam' 'sgd'
    
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)  
 
 
loss_list.append(history.history['loss'][-1])
  
print('\n\n\nlr : ', lr, '   ', loss, ': ', loss_list[0])
  
casename = 'lr_'+str(lr)+'-lrdecay_'+str(lrdecay)+'-act_'+actname+'-opt_'+optname+'-hiddenlayers_'+layers+'-neurouns_'+neurons+'-batchnorm_'+batchnorm+'-dropout_'+drop+'-batchsize_'+str(batch_size)+'-epochs_'+str(epochs)+'-inputs_'+str(X.shape[1])
 
# Summarize history for loss
figure = plt.figure(figsize=(10., 8.), dpi=300)
plt.semilogy(history.history['loss'], 'k-')
plt.semilogy(history.history['val_loss'], 'k:')
plt.ylabel('Loss', font_normal, rotation='vertical')
plt.xlabel('Epoch', font_normal)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
figname = './out/Loss_'+loss+'-'+casename+'.png'
plt.savefig(figname, transparent=True)
figure.clear()
plt.close(figure)

# Summarize history for mape
figure = plt.figure(figsize=(10., 8.), dpi=300)
plt.semilogy(history.history['mean_absolute_percentage_error'], 'k-')
plt.semilogy(history.history['val_mean_absolute_percentage_error'], 'k:')
plt.ylabel('MAPE', font_normal, rotation='vertical')
plt.xlabel('Epoch', font_normal)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
figname = './out/MAPE-'+casename+'.png'
plt.savefig(figname, transparent=True)
figure.clear()
plt.close(figure)

train_scores = model.evaluate(X_train, Y_train) 
print("\n%s - %s: %.6f - %s: %.6f%%" % ('Train', model.metrics_names[2], train_scores[2], model.metrics_names[3], train_scores[3]))

X_test = scaler.transform(X_test)
test_scores = model.evaluate(X_test, Y_test) 
print("\n%s - %s: %.6f - %s: %.6f%%" % ('Test', model.metrics_names[2], test_scores[2], model.metrics_names[3], test_scores[3]))
         
print('\n\n')