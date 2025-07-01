# %%
import pickle

# %%
with open("protein_embeddings_1.pkl", "rb") as f:
    dict1 = pickle.load(f)

# %%
with open("protein_embeddings_2.pkl", "rb") as f:
    dict2 = pickle.load(f)

# %%
merged_dict = {**dict1, **dict2}

with open("protein_embeddings_all.pkl", "wb") as f:
    pickle.dump(merged_dict, f)
    
    
# %%
