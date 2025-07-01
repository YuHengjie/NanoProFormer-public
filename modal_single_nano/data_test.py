# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# %%
test_df = pd.read_csv("test_data.csv",keep_default_na=False, na_values=[''])
test_df

# %%
threshold = 0.001
test_df['Affinity_Category'] = test_df['RPA'].apply(lambda x: 0 if x <= threshold else 1)
test_df

# %%
test_df.to_csv('test_data_binary.csv', index=False)

# %%
lambda_ = np.load('lambda_.npy', allow_pickle=True)
lambda_

# %%
def apply_boxcox(data, lambda_):
    if lambda_ == 0:
        return np.log(data)
    else:
        return (np.power(data, lambda_) - 1) / lambda_

# %%
test_df = test_df[test_df['Affinity_Category']==1]
test_df

# %%
test_df['Box-Cox RPA'] = apply_boxcox(test_df['RPA'], lambda_)
test_df

# %%
test_df.to_csv('test_data_reg.csv', index=False)


# %%
