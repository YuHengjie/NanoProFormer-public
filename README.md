# NanoProFormer

This repository contains the source code for model construction and explanation in [**A million-scale dataset and generalizable foundation model for nanomaterial-protein interactions**]
(https://arxiv.org/abs/2507.14245). If you utilize the code or algorithms presented here, your citation would be greatly appreciated (arXiv:2507.14245).

## **Folder Contents**

* **`data`**: This folder is designated for storing datasets.
* **`X_embedding`**: Contains scripts for obtaining embedding vectors for structured text data.
* **`protein_embedding`**: Contains scripts for obtaining embedding vectors for protein sequences.
* **`modal_fusion`**: Holds the code for model building based on the imputed dataset.
* **`modal_fusion_hybrid`**: Contains the code for model building based on the hybrid dataset.
* **`modal_single_nano`**: Includes scripts for model building based solely on the tabular modality.
* **`modal_single_pro`**: Includes scripts for model building based solely on the protein modality.
* **`X_embedding_4_explain`**: Dedicated to obtaining embedding vectors specifically for model explanation.
* **`model_explain`**: Contains the code for model explanation.