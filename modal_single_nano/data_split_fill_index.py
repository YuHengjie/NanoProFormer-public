# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# %%
df = pd.read_csv("../data/dataset_curated_fill.csv",keep_default_na=False, na_values=[''])
df

# %% 
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True) 

train_val_df, test_df = train_test_split(df_shuffled, test_size=0.1, random_state=42)

# %% 
train_val_df['RPA_binned'], bin_edges = pd.qcut(train_val_df['RPA'], q=150, labels=False,
                                     duplicates='drop', retbins=True)
train_val_df

# %%
print("\nBin Edges:")
print(bin_edges)

# %%
train_df, val_df = train_test_split(train_val_df, test_size=0.111111, stratify=train_val_df['RPA_binned'], random_state=42)
train_df

# %%
train_df = train_df.drop(columns=['RPA_binned'])
val_df = val_df.drop(columns=['RPA_binned'])
train_df

# %%
print("Original DataFrame shape:", df.shape)
print("Train DataFrame shape:", train_df.shape)
print("Validation DataFrame shape:", val_df.shape)
print("Test DataFrame shape:", test_df.shape)

# %%
train_index = train_df.index.to_numpy()
val_index = val_df.index.to_numpy()
test_index = test_df.index.to_numpy()

np.save('train_index.npy', train_index)
np.save('validation_index.npy', val_index)
np.save('test_index.npy', test_index)

# %%
