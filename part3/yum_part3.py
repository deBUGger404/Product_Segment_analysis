import pandas as pd
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Menu Items.csv")

#from google.colab import drive
#drive.mount('/content/drive')

df.head()

df = df[(~df['Description'].isna()) & (~df['Item'].isna())].drop_duplicates()
df['Price'] = df['Price'].str.replace("$","").astype(float)

df.dtypes

df['max_price'] = df.groupby('Restaurant')['Price'].transform('max')
df['min_price'] = df.groupby('Restaurant')['Price'].transform('min')
df['avg_price'] = df.groupby('Restaurant')['Price'].transform('mean')

df_maxprice = df[df['Price']==df['max_price']]
df_minprice = df[df['Price']==df['min_price']]

## Items which are priced high wrt to other menu items for each restaurant
df_maxprice.to_csv("high_prices_items.csv")

## Items which are priced low wrt to other menu items for each restaurant
df_minprice.to_csv("low_prices_items.csv")

df1 = df.drop_duplicates('Restaurant')

import seaborn as sns
sns.boxplot(df1['avg_price'])

## Restaurants which have higher prices wrt other restaurants
df1[df1['avg_price']>30]

## Restaurants which have lower prices wrt other restaurants
df1[df1['avg_price']<5]

# Scatter plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(df1['avg_price'], df1['Restaurant'])
 
# x-axis label
ax.set_xlabel('avg_price')
 
# y-axis label
ax.set_ylabel('Res')
plt.show()

