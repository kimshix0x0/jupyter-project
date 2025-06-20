# %%
import pandas as pd

# %%
data = pd.read_csv('eco-dataset.csv')

# %%
data

# %%
print(data)

# %%
print(data.tail)

# %%
data.tail

# %%
data.describe()

# %%
data.info()

# %%
import numpy as np

# %%
print(data.isnull().sum())

# %%
data['GDP Growth Rate (%)'] = data['GDP Growth Rate (%)'].replace(np.nan,data['GDP Growth Rate (%)'].mean())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import plotly.express as px

# %%
x= np.array([3,5,3,6])
y= np.array([9,25,9,36])            
            

# %%
plt.plot(x,y,'o')
plt.show()

# %%
y = np.array([35,24,31,43,26,36,37])

# %%
plt.plot(y)
plt.show()

# %%
plt.pie(y)
plt.show()

# %%
plt.hist(y)
plt.show()

# %%



