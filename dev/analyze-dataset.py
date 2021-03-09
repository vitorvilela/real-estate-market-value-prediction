import pandas as pd
from pandas.io.json import json_normalize

# LabelEncoder maps a set of string to a set of integers. Although the magnitude of these numbers has no meaning, the model can misunderstood it.
# It is useful to encode output data; and also for categorical feature visualization using scatter plot, for instance.
from sklearn.preprocessing import LabelEncoder

# The OneHotEncoder avoids the string to numeric mapping to have any magnitude meaning
# It creates N columns associated to the N classes of the encoded feature
#from sklearn.preprocessing import OneHotEncoder


from matplotlib import pyplot as plt
from matplotlib import cm as cm
from pandas import set_option

font_normal = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontsize'   : 18 }

font_italic = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontstyle'  : 'italic',
                'fontsize'   : 18 }

# O coeficiente de correlação de Pearson, rxy ∈ [−1, +1], fornece uma medida da relação linear entre duas variáveis
# O coeficiente de correlação de Spearman, rs ∈ [−1, +1], por sua vez, indica se duas variáveis são monotônicas, independentemente da relação linear. 
# Adicionalmente, o coeficiente de Spearman é menos sensível aos outliers de uma amostra do que o coeficiente de Pearson.

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
dataframe_group = dataframe_clean.loc[dataframe_clean['pricingInfos.price'] < 1.e6] # >= 1.e6]
# below million 51.120
# above million 12.813

dataframe = dataframe_group

#How to inspect the dataframe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #print([c for c in dataframe.columns])
  print(dataframe.head())
  #print(dataframe.dtypes)
  #print(dataframe.loc[:, ['pricingInfos.businessType']])
  #print(dataframe.groupby('pricingInfos.businessType').size())

## Summarize the dataset by descriptive statistics
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   
  set_option('precision', 4) 
   
  #Confirming the dimensions of the dataframe.
  print('\ndataframe shape\n', dataframe.shape, '\n')

  # Look at the data types of each attribute. Are all attributes numeric?
  print('dataframe types\n', dataframe.dtypes, '\n')

  # Take a peek at the first 5 rows of the data. Look at the scales for the attributes.
  print('dataframe head\n', dataframe.head(2), '\n')

  # Summarize the distribution of each attribute (e.g. min, max, avg, std). How different are the attributes?
  print('dataframe describe\n', dataframe.describe(), '\n')
    
  # Have attributes a strong correlation (e.g. > 0.70 or < -0.70) among them? and with the outputs?
  print('dataframe correlation - ', corr_method, '\n', dataframe.corr(method=corr_method), '\n')


# Data visualizations
# Think about: 
# - Feature selection and removing the most correlated attributes
# - Normalizing the dataframe to reduce the effect of differing scales
# - Standardizing the dataframe to reduce the effects of differing distributions

names = ['pricingInfos.price']
for n in names:
  
  # Histograms
  # Get a sense of the data distributions (e.g., uniform, exponential, bimodal, normal).
  figure = plt.figure(figsize=(10., 8.), dpi=300)
  dataframe.loc[:,n].plot(kind='hist', bins=10, alpha=0.5, color='green',fontsize=16, legend=None) # logx=True, logy=True
  plt.xlabel(n, font_italic)
  plt.ylabel('Count', font_italic, rotation='vertical')
  plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  #plt.show()
  plt.tight_layout()
  plt.savefig('./out/hist-'+n+'.png', transparent=True)
  figure.clear()
  plt.close(figure)
 
  # Box and whisker plots
  # Boxplots summarize the distribution of each attribute, drawing a line for the median and a box around the 25th and 75th percentiles.
  # The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate outlier values.
  # Outliers: values that are 1.5 times greater than the size of spread of the middle 50% of the data (i.e. 75th - 25th percentile).
  figure = plt.figure(figsize=(10., 8.), dpi=300)
  dataframe.loc[:,n].plot(kind='box', color='k', fontsize=16, legend=None)  
  plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
  #plt.show()
  plt.tight_layout()
  plt.savefig('./out/box-'+n+'.png', transparent=True)
  figure.clear()
  plt.close(figure)



with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  
  print(dataframe['address.neighborhood'], '\n')
  
  le = LabelEncoder()
  le.fit(dataframe['address.neighborhood'])

  print(list(le.classes_), '\n')
  
  dataframe['encoded.address.neighborhood'] = le.transform(dataframe['address.neighborhood'])
  # len(dataframe['encoded.address.neighborhood']) = 72.241  
       
i = 'encoded.address.neighborhood'   
o = 'pricingInfos.price'  

#print(type(dataframe['encoded.address.neighborhood']))

#labels = le.inverse_transform(dataframe['encoded.address.neighborhood'])
        
figure = plt.figure(figsize=(10., 8.), dpi=300)
dataframe.loc[:,[i,o]].plot(kind='scatter', x=i, y=o, s=None, c='k', logy=True, fontsize=16) # logx=True, logy=True, fontsize=16)
plt.xlabel(i, font_normal)
#plt.xlabel(i, labels, font_normal, rotation='vertical')
plt.ylabel(o, font_normal, rotation='vertical')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#plt.show()
plt.tight_layout()
plt.savefig('./out/scatter-'+i+'-'+o+'.png', transparent=True)
figure.clear()
plt.close(figure)  