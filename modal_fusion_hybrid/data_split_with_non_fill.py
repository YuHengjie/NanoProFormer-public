# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

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

plt.figure(figsize=(4.2, 2.6))

counts, bins, patches = plt.hist(diff_values, bins=range(int(min(diff_values)) - 1, 
                                                         int(max(diff_values)) + 2), align='mid', 
                                 color='#778899', edgecolor='#778899', rwidth=0.7)
plt.xlabel('Number of filled values')
plt.ylabel('Sample count')

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.xlim(min(diff_values), max(diff_values))

plt.xticks(bin_centers, [int(x) for x in bin_centers])
plt.tight_layout()

plt.savefig('number_of_filled.png', dpi=600,
            bbox_inches='tight', facecolor='none',
            transparent=True)  

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
df_all = pd.concat([df_non_fill, df_fill], axis=0)
df_all

# %%
df_all.columns[1:30]

# %%
selected_columns = df_all.columns[1:30]

data = {
    'Feature': selected_columns,
    'Unknown_All': [df_all[col].astype(str).str.lower().str.contains('unknown').sum() for col in selected_columns],
    'Unknown_Fill': [df_fill[col].astype(str).str.lower().str.contains('unknown').sum() for col in selected_columns]
}

df_unknown = pd.DataFrame(data)

df_unknown['Difference'] = df_unknown['Unknown_All'] - df_unknown['Unknown_Fill']
df_unknown['Percentage_Difference'] = (df_unknown['Difference'] / df_all.shape[0]) * 100

df_unknown

# %%
df_unknown['Feature'] = df_unknown['Feature'].str.replace("℃", "°C")


# %%
plt.figure(figsize=(7, 3))
sns.barplot(x='Feature', y='Percentage_Difference', data=df_unknown, color='#778899')
#plt.xlabel('Column')
plt.ylabel('Filling (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('percentaga_of_filled.png', dpi=600,
            bbox_inches='tight', facecolor='none',
            transparent=True)  