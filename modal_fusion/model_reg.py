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
with open("../protein_embedding/protein_embeddings_all.pkl", "rb") as f:
    pro_embed_dict = pickle.load(f)
    
# %%
first_key = next(iter(pro_embed_dict))  
array_shape = pro_embed_dict[first_key].shape
print(f"Array shape for {first_key}: {array_shape}")

# %%
x_embed_data = np.load("../X_embedding/x_embeddings_fill.npy")  
x_embed_data.shape

# %%
train_df = pd.read_csv("train_data_reg.csv",keep_default_na=False, na_values=[''])
val_df = pd.read_csv("validation_data_reg.csv",keep_default_na=False, na_values=[''])
train_df

# %%
train_rpa = train_df['RPA'].values
val_rpa = val_df['RPA'].values
rpa_values = np.concatenate([train_rpa, val_rpa])  

# %%
plt.hist(rpa_values, bins=100, edgecolor='black') 
plt.title('Histogram of RPA Values')
plt.xlabel('RPA Value')
plt.ylabel('Frequency')
plt.show()

# %%
rpa_boxcox, lambda_ = boxcox(rpa_values) 
plt.hist(rpa_boxcox, bins=100, edgecolor='black') 
plt.title('Histogram of rpa_boxcox Values')
plt.xlabel('rpa_boxcox Value')
plt.ylabel('Frequency')
plt.show()

# %%
predicted_original = inv_boxcox(rpa_boxcox, lambda_)
predicted_original

# %%
np.save('lambda_.npy', lambda_)

# %%
def apply_boxcox(data, lambda_):
    if lambda_ == 0:
        return np.log(data)
    else:
        return (np.power(data, lambda_) - 1) / lambda_
    
# %%
train_df['Box-Cox RPA'] = apply_boxcox(train_df['RPA'], lambda_)
val_df['Box-Cox RPA'] = apply_boxcox(val_df['RPA'], lambda_)
train_df


# %%
class EmbeddingPairDataset(Dataset):
    def __init__(self, df, x_embed_data, pro_embed_dict):
        self.df = df.reset_index(drop=True)
        self.x_embed_data = x_embed_data
        self.pro_embed_dict = pro_embed_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_embed = self.x_embed_data[int(row["x_index"])]
        pro_embed = self.pro_embed_dict[row["Accession"]]
        rpa = row["Box-Cox RPA"]
        return torch.tensor(x_embed, dtype=torch.float32), \
               torch.tensor(pro_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32)

train_dataset = EmbeddingPairDataset(train_df, x_embed_data, pro_embed_dict)
val_dataset = EmbeddingPairDataset(val_df, x_embed_data, pro_embed_dict)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# %%
class CrossAttentionRegressor(nn.Module):
    def __init__(self, x_dim, pro_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.x_mlp = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pro_mlp  = nn.Sequential(
            nn.Linear(pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
         
        self.attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout/2  
        )
        
        self.regressor  = nn.Sequential(
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
        for module in [self.x_mlp, self.pro_mlp,  self.regressor]: 
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight,  nonlinearity='relu')
                    nn.init.constant_(layer.bias,  0.1)
 
    def forward(self, x_embed, pro_embed):
        x_feat = self.x_mlp(x_embed)        
        pro_feat = self.pro_mlp(pro_embed)   
        
        x_feat = x_feat.unsqueeze(1) 
        pro_feat = pro_feat.unsqueeze(1) 
        
        attn_out, _ = self.attn( 
            query=x_feat,
            key=pro_feat,
            value=pro_feat,
            need_weights=False 
        )
        
        fused_feature = x_feat + attn_out 
        fused_feature = fused_feature.squeeze(1)   
        
        reg_output  = self.regressor(fused_feature).squeeze(-1)   
        return reg_output
    
# %%
x_dim = x_embed_data.shape[1]
pro_dim = next(iter(pro_embed_dict.values())).shape[0]

model = CrossAttentionRegressor(x_dim, pro_dim).to(device)

# %%
def print_model_params_count(model):
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if num_params > 0:  
            print(f"Layer: {name}, Number of parameters: {num_params}")

print_model_params_count(model)

# %%
def train_and_validate(model, criterion, train_loader, val_loader, epochs=10, lr=1e-4, log_interval=1):
    """
    Trains and validates the model for regression.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimizer.
        log_interval (int): Interval (in epochs) for printing logs.
    """
    
    optimizer = torch.optim.AdamW(model.parameters(),  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3, patience=5,)
    best_r2 = 0
    train_loss_all = []
    val_loss_all = []
    val_mae_all = []
    val_mse_all = []
    val_r2_all = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, pro, y in train_loader:
            x, pro, y = x.to(device), pro.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x, pro)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_targets = []
        total_val_loss = 0.0
        
        with torch.no_grad():
            for x, pro, y in val_loader:
                x, pro, y = x.to(device), pro.to(device), y.to(device)

                outputs = model(x, pro)
                loss = criterion(outputs, y)  
                total_val_loss += loss.item()  
                
                preds = outputs.detach().cpu()
                all_preds.append(preds)
                all_targets.append(y.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_targets = torch.cat(all_targets).numpy().flatten()

        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        if epoch % log_interval == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {avg_val_loss:.4f} | "
                f"Validation MSE: {mse:.4f} | "
                f"Validation MAE: {mae:.4f} | "
                f"R2: {r2:.4f} | "
                f"Current LR: {current_lr:.6f}"
            )
        
        train_loss_all.append(avg_train_loss)
        val_loss_all.append(avg_val_loss)
        val_mse_all.append(mse)
        val_mae_all.append(mae)
        val_r2_all.append(r2)
        
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), "best_model_reg.pt")
            print(f"Best model saved at epoch {epoch} with R2: {r2:.4f}")
        
        scheduler.step(r2) 
    
    save_dir = './train_result_reg'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'train_loss_all.npy'), train_loss_all)
    np.save(os.path.join(save_dir, 'val_loss_all.npy'), val_loss_all)
    np.save(os.path.join(save_dir, 'val_mse_all.npy'), val_mse_all)
    np.save(os.path.join(save_dir, 'val_mae_all.npy'), val_mae_all)
    np.save(os.path.join(save_dir, 'val_r2_all.npy'), val_r2_all)
    
    print(f"Training complete. Best validation R2: {best_r2:.4f}")

# %%
epochs = 200  
learning_rate = 1e-3  
criterion = torch.nn.SmoothL1Loss()

train_and_validate(model, criterion, train_loader, val_loader, epochs=epochs, lr=learning_rate)

# %%
