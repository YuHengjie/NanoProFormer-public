# %%
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
df = pd.read_excel("protein_seq_20250418.xlsx", index_col=0)
df = df.iloc[0:17000,:] 
df

# %% 
model_path = "/yuhengjie/backup/pretrainedmodel/esm2_t36_3B_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model = model.to(device)
model.eval()

# %%
batch_size = 8  

embedding_dict = {}

for i in tqdm(range(0, len(df), batch_size), desc="Encoding"):
    batch_df = df.iloc[i:i+batch_size]
    accessions = batch_df["Accession"].tolist()
    sequences = batch_df["Sequence"].tolist()

    inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        attention_mask = inputs["attention_mask"]
        embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    for acc, emb in zip(accessions, embeddings.cpu()):
        emb_float32 = emb.to(torch.float32)
        embedding_dict[acc] = emb_float32.numpy()
        
    del inputs
    del outputs
    del last_hidden
    del embeddings
    torch.cuda.empty_cache()  

# %%
with open("protein_embeddings_1.pkl", "wb") as f:
    pickle.dump(embedding_dict, f)
    