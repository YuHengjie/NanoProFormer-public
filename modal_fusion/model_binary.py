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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime

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
train_df = pd.read_csv("train_data_binary.csv",keep_default_na=False, na_values=[''])
val_df = pd.read_csv("validation_data_binary.csv",keep_default_na=False, na_values=[''])
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
        rpa = row["Affinity_Category"]
        return torch.tensor(x_embed, dtype=torch.float32), \
               torch.tensor(pro_embed, dtype=torch.float32), \
               torch.tensor(rpa, dtype=torch.float32)

train_dataset = EmbeddingPairDataset(train_df, x_embed_data, pro_embed_dict)
val_dataset = EmbeddingPairDataset(val_df, x_embed_data, pro_embed_dict)

batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)


# %%
class CrossAttentionClassifier(nn.Module):
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
         
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, hidden_dim//16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//16, 1) 
        )
 
        self._init_weights()
 
    def _init_weights(self):
        for module in [self.x_mlp, self.pro_mlp,  self.classifier]: 
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight,  nonlinearity='relu')
                    nn.init.constant_(layer.bias,  0.1)
 
    def forward(self, x_embed, pro_embed):
        x_feat = self.x_mlp(x_embed)        # [B, 1024]
        pro_feat = self.pro_mlp(pro_embed)   # [B, 1024]
        
        x_feat = x_feat.unsqueeze(1) 
        pro_feat = pro_feat.unsqueeze(1) 
        
        attn_out, _ = self.attn( 
            query=x_feat,
            key=pro_feat,
            value=pro_feat,
            need_weights=False 
        )
        
        fused_feature = x_feat + attn_out  # [B, 1, 1024]
        fused_feature = fused_feature.squeeze(1)   # [B, 1024]
        
        logits = self.classifier(fused_feature).squeeze(-1)   # [B]
        return logits 
    
# %%
x_dim = x_embed_data.shape[1]
pro_dim = next(iter(pro_embed_dict.values())).shape[0]

model = CrossAttentionClassifier(x_dim, pro_dim).to(device)

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
    Trains and validates the model for binary classification.

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
    best_auroc = 0.0
    train_loss_all = []
    val_loss_all = []
    val_auroc_all = []
    val_f1_all = []
    f1_threshold_all = []
    
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
        all_probs = []
        all_targets = []
        total_val_loss = 0.0
        
        with torch.no_grad():
            for x, pro, y in val_loader:
                x, pro, y = x.to(device), pro.to(device), y.to(device)

                outputs = model(x, pro)
                loss = criterion(outputs, y)  
                total_val_loss += loss.item()  
                
                probs = torch.sigmoid(outputs).detach().cpu()
                all_probs.append(probs)
                all_targets.append(y.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        all_probs = torch.cat(all_probs).numpy().flatten()
        all_targets = torch.cat(all_targets).numpy().flatten()

        thresholds = np.arange(0.1, 0.91, 0.01)
        best_f1 = 0
        best_threshold = 0.5

        for t in thresholds:
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_targets, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                np.save('threshold_binary.npy', best_threshold)
                
        preds = (all_probs > best_threshold).astype(int)

        precision = precision_score(all_targets, preds)
        recall = recall_score(all_targets, preds)
        f1 = f1_score(all_targets, preds)
        auc_score = roc_auc_score(all_targets, all_probs)

        if epoch % log_interval == 0:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {avg_val_loss:.4f} | "
                f"Validation AUROC: {auc_score:.4f} | "
                f"Best Threshold: {best_threshold:.2f} | "
                f"F1: {f1:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"Current LR: {current_lr:.6f}"
            )
        
        train_loss_all.append(avg_train_loss)
        val_loss_all.append(avg_val_loss)
        val_auroc_all.append(auc_score)
        val_f1_all.append(f1)
        f1_threshold_all.append(best_threshold)
        
        if auc_score > best_auroc:
            best_auroc = auc_score
            torch.save(model.state_dict(), "best_model_binary.pt")
            print(f"Best model saved at epoch {epoch} with AUROC: {best_auroc:.4f}")
        
        scheduler.step(auc_score)  
    
    save_dir = './train_result_binary'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'train_loss_all.npy'), train_loss_all)
    np.save(os.path.join(save_dir, 'val_loss_all.npy'), val_loss_all)
    np.save(os.path.join(save_dir, 'val_auroc_all.npy'), val_auroc_all)
    np.save(os.path.join(save_dir, 'val_f1_all.npy'), val_f1_all)
    np.save(os.path.join(save_dir, 'f1_threshold_all.npy'), f1_threshold_all)
    
    print(f"Training complete. Best validation AUROC: {best_auroc:.4f}")

# %%
class_counts = train_df['Affinity_Category'].value_counts()
 
num_neg = class_counts.get(0,  0) 
num_pos = class_counts.get(1,  0) 

pos_weight = torch.tensor([num_neg/num_pos*2.0], device=device) if num_pos > 0 else torch.tensor([1.0],  device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# %%
epochs = 150  
learning_rate = 1e-3 

train_and_validate(model, criterion, train_loader, val_loader, epochs=epochs, lr=learning_rate)

# %%
