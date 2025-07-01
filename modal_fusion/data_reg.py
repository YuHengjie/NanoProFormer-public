# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# %%
train_df = pd.read_csv("train_data.csv",keep_default_na=False, na_values=[''])
val_df = pd.read_csv("validation_data.csv",keep_default_na=False, na_values=[''])
train_df

# %%
train_rpa = train_df['RPA'].values
val_rpa = val_df['RPA'].values
rpa_values = np.concatenate([train_rpa, val_rpa]) 

# %%
threshold = 0.001
count_below_threshold = (rpa_values < threshold).sum()  
total_count = len(rpa_values)  
percentage_below_threshold = count_below_threshold / total_count * 100  

print(f"Number of samples with RPA < threshold: {count_below_threshold}")
print(f"Total number of samples: {total_count}")
print(f"Percentage of samples with RPA < threshold: {percentage_below_threshold:.2f}%")

# %%
train_df = train_df[train_df['RPA'] > threshold]
val_df = val_df[val_df['RPA'] > threshold]
train_df

# %%
train_rpa = train_df['RPA'].values
val_rpa = val_df['RPA'].values
rpa_values = np.concatenate([train_rpa, val_rpa])  

# %%
# 绘制直方图
plt.hist(rpa_values, bins=100, edgecolor='black') 
plt.title('Histogram of RPA Values')
plt.xlabel('RPA Value')
plt.ylabel('Frequency')
plt.show()

# %%
# 方法 1: scipy 的 boxcox
rpa_boxcox, lambda_ = boxcox(rpa_values) 
plt.hist(rpa_boxcox, bins=100, edgecolor='black')  
plt.title('Histogram of rpa_boxcox Values')
plt.xlabel('rpa_boxcox Value')
plt.ylabel('Frequency')
plt.show()

# %%
pt = PowerTransformer(method='box-cox', standardize=True)
rpa_boxcox = pt.fit_transform(rpa_values.reshape(-1, 1))
plt.hist(rpa_boxcox, bins=100, edgecolor='black') 
plt.title('Histogram of rpa_boxcox Values')
plt.xlabel('rpa_boxcox Value')
plt.ylabel('Frequency')
plt.show()

# %%
qt = QuantileTransformer(output_distribution='normal')
rpa_transformed = qt.fit_transform(rpa_values.reshape(-1, 1))
plt.hist(rpa_transformed, bins=100, edgecolor='black')  
plt.title('Histogram of rpa_transformed Values')
plt.xlabel('rpa_transformed Value')
plt.ylabel('Frequency')
plt.show()

# %% 
rpa_log = np.log1p(rpa_values)
plt.hist(rpa_log, bins=100, edgecolor='black') 
plt.title('Histogram of log RPA Values')
plt.xlabel('rpa_boxcox Value')
plt.ylabel('Frequency')
plt.show()

# %%
train_df.to_csv('train_data_reg.csv', index=False)
val_df.to_csv('validation_data_reg.csv', index=False)

# %%
