import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from pandas.io.json import json_normalize


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn.dummy import DummyRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

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
dataframe_group = dataframe_clean.loc[dataframe_clean['pricingInfos.price'] >= 1.e6] # >= 1.e6]
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

# Split into input (X) and output (Y) variables

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

models = [DummyRegressor(), LinearRegression(), ElasticNet(), DecisionTreeRegressor(), ExtraTreesRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]
X_validation = X_test
Y_validation = Y_test

degrees = (1,)
for degree in degrees:
  
  print('\nDegree: ', degree)

  polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

  train_input = polynomial_features.fit_transform(X_train)
  scaler = StandardScaler().fit(train_input)
  rescaledX = scaler.transform(train_input)
  
  for model in models:
  
    name = repr(model).split('(')[0]   
    model.fit(rescaledX, Y_train)

    val_input = polynomial_features.fit_transform(X_validation)
    rescaledValidationX = scaler.transform(val_input)

    predictions = model.predict(rescaledValidationX)
    
    mae = mean_absolute_error(Y_validation, predictions)
    mape = np.abs(100*(predictions-Y_validation)/(Y_validation+EPS))
    mse = mean_squared_error(Y_validation, predictions)
    r2 = r2_score(Y_validation, predictions)
                               
    print('\n' + name)
    print('\nMAE [-]:')
    print(mae.mean())
    print('\nMAPE [%]:')
    print(mape.mean())
    print('\nRMSE [-]:')
    print(np.sqrt(mse))
    print('\nR2 [-]:')
    print(r2)
    print('\nMSE [-]:')
    print(mse)


# Evaluate Algorithms with cross-validation

# Test options and evaluation metric
# 10-fold cross-validation is a good standard test when dataset is not too small (e.g. 500)
num_folds = 10
print('\nCross validation')
print('num_folds ' + str(num_folds))

# 'neg_mean_squared_error' MSE will give a gross idea of how wrong all predictions are (0 is perfect)
# others: 'r2', 'explained_variance'
#scores = [('R2', 'r2'), ('MSE', 'neg_mean_squared_error'), ('MAE', 'neg_mean_absolute_error')]
scores = [('MAE', 'neg_mean_absolute_error')]

degrees = (1,)

print('\nStandardize and Using Polynomial Features') 

for degree in degrees:
    
  print('\nDegree %i' % degree)
           
  polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

  pipelines = []
  pipelines.append(('DUMMY', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('Dummy', DummyRegressor())])))
  pipelines.append(('LR', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('LR', LinearRegression())])))
  pipelines.append(('EN', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('EN', ElasticNet())])))
  pipelines.append(('CART', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
  
  # Ensemble
  pipelines.append(('RF', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
  pipelines.append(('ET', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))
  pipelines.append(('AB', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
  pipelines.append(('GBM', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
  
  kfold = KFold(n_splits=num_folds, random_state=seed)

  for score, scoring in scores:
      
    print('\nScore %s' % score)
    
    results = []
    names = []
      
    for name, model in pipelines:     
          
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      results.append(cv_results)                                      
      names.append(name)

      msg = "%s: %e (%e)" % (name, cv_results.mean(), cv_results.std())
      print(msg)           
                        
    figure = plt.figure(figsize=(10., 10.), dpi=300)
    ax = figure.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, font_italic, rotation='vertical')
    plt.ylabel(score, font_italic, rotation='vertical')
    plt.savefig('./out/degree-'+str(degree)+'-'+score+'.png', transparent=True)
    figure.clear()
    plt.close(figure)