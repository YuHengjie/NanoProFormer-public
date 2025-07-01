# %%
# %%
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy.stats import boxcox
from datetime import datetime 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from scipy.stats import boxcox
from scipy.special import inv_boxcox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                     
    torch.cuda.manual_seed(seed)                

set_seed(42)

# %%
x_embed_data_fill = np.load("../X_embedding/x_embeddings_fill.npy")  
x_embed_data_fill.shape

# %%
x_embed_data_non_fill = np.load("../X_embedding/x_embeddings_non_fill.npy")  
x_embed_data_non_fill.shape


# %%
test_df_all = pd.read_csv("test_data_reg.csv",keep_default_na=False, na_values=[''])
test_df_all

# %%
class EmbeddingPairDataset(Dataset):
    def __init__(self, df, x_embed_data_fill, x_embed_data_non_fill,):
        self.df = df.reset_index(drop=True)
        self.x_embed_data_fill = x_embed_data_fill
        self.x_embed_data_non_fill = x_embed_data_non_fill

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if row['Fill status'] == 1:
            x_embed = self.x_embed_data_fill[int(row["x_index"])]
        else:
            x_embed = self.x_embed_data_non_fill[int(row["x_index"])]
                
        rpa = row["Box-Cox RPA"]
        
        return torch.tensor(x_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32)

# %%
class SingleModalRegressor(nn.Module):
    def __init__(self, x_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2  
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1),
            nn.Identity()  
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.x_mlp, self.regressor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x_embed):
        x_feat = self.x_mlp(x_embed)      

        x_feat = x_feat.unsqueeze(1)

        attn_out, _ = self.attn(
            query=x_feat,
            key=x_feat,
            value=x_feat,
            need_weights=False
        )

        fused_feature = x_feat + attn_out    
        fused_feature = fused_feature.squeeze(1)  

        reg_output = self.regressor(fused_feature).squeeze(-1)  
        return reg_output

# %%
def test_model_on_testset(model, test_loader,):
    """
    Evaluates the trained classification model on the test dataset.

    Args:
        model (nn.Module): Trained classification model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to use for evaluation.

    Returns:
        dict: A dictionary containing test metrics.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x,)
            preds = outputs.detach().cpu()
            all_preds.append(preds)
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print("=== Test Set Evaluation ===")
    print(f"R2: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        'R2': r2,
        'MSE': mse,
        'MAE': mae,
        'Preds': all_preds
    }
    
# %% 
test_df = test_df_all.copy()

# %%
test_dataset = EmbeddingPairDataset(test_df, x_embed_data_fill, x_embed_data_non_fill,)

batch_size = 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# %%
x_dim = x_embed_data_fill.shape[1]

regressor = SingleModalRegressor(x_dim,)
regressor.load_state_dict(torch.load("best_model_reg.pt"))
regressor.eval()

# %%
metrics = test_model_on_testset(regressor, test_loader)
metrics

