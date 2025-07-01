# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %% 
df_fill = pd.read_csv("../data/dataset_curated_fill.csv",keep_default_na=False, na_values=[''])
df_fill

# %%
df_fill['Fill status'] = 1
df_fill

# %% 
loaded_train_index = np.load('train_index.npy')
loaded_val_index = np.load('validation_index.npy')
loaded_test_index = np.load('test_index.npy')

# %%
df_fill_train = df_fill.iloc[loaded_train_index]
df_fill_val = df_fill.iloc[loaded_val_index]
df_fill_test = df_fill.iloc[loaded_test_index]

# %% 
df_non_fill = pd.read_csv("../data/dataset_curated_non_fill.csv",keep_default_na=False, na_values=[''])
df_non_fill

# %%
df_non_fill['Fill status'] = 0
df_non_fill

# %% 
loaded_train_index = np.load('train_index.npy')
loaded_val_index = np.load('validation_index.npy')
loaded_test_index = np.load('test_index.npy')

# %%
df_non_fill_train = df_non_fill.iloc[loaded_train_index]
df_non_fill_val = df_non_fill.iloc[loaded_val_index]
df_non_fill_test = df_non_fill.iloc[loaded_test_index]


# %% 
non_value_count = pd.DataFrame(index=df_fill.index, columns=['non_value_count_fill', 'non_value_count_non_fill', 'non_value_count_diff', 'non_value_count_diff_ratio'])

non_value_count['non_value_count_fill'] = (df_fill.iloc[:, 1:30] == 'Unknown').sum(axis=1)

non_value_count['non_value_count_non_fill'] = (df_non_fill.iloc[:, 1:30] == 'Unknown').sum(axis=1)

non_value_count['non_value_count_diff'] = non_value_count['non_value_count_non_fill'] - non_value_count['non_value_count_fill']
non_value_count['non_value_count_diff_ratio'] = non_value_count['non_value_count_diff'] / 29 * 100

non_value_count

# %%
diff_values = non_value_count['non_value_count_diff'].values

plt.figure(figsize=(6, 4))

counts, bins, patches = plt.hist(diff_values, bins=range(int(min(diff_values)) - 1, int(max(diff_values)) + 2), align='mid', edgecolor='black', rwidth=0.8)
plt.title('Histogram of Unknown Value number difference')
plt.xlabel('Unknown Value number difference')
plt.ylabel('Frequency')

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.xticks(bin_centers, [int(x) for x in bin_centers])
plt.tight_layout()

plt.savefig('na_value_number_diff.png', dpi=600)  

# %%
df_train = pd.concat([df_non_fill_train, df_fill_train], axis=0)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)  
df_val = pd.concat([df_non_fill_val, df_fill_val], axis=0)
df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True) 
df_test = pd.concat([df_non_fill_test, df_fill_test], axis=0)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)  
df_train

# %%
print("Train DataFrame shape:", df_train.shape)
print("Validation DataFrame shape:", df_val.shape)
print("Test DataFrame shape:", df_test.shape)

# %%
df_train.to_csv('train_data.csv', index=False)
df_val.to_csv('validation_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)

# %%
