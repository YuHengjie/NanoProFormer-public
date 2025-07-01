# %%
import pandas as pd
import numpy as np

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
train_df['Affinity_Category'] = train_df['RPA'].apply(lambda x: 0 if x <= threshold else 1)
val_df['Affinity_Category'] = val_df['RPA'].apply(lambda x: 0 if x <= threshold else 1)

train_df

# %%
train_df.to_csv('train_data_binary.csv', index=False)
val_df.to_csv('validation_data_binary.csv', index=False)

# %%
