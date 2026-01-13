# MTL-GCN-AqTox
The files related to the paper published in Chemical Research in Toxicology:
"Multi-Task Graph Convolutional Network Model with Improved Performance and Attention Mechanisms for Identifying Chemicals with Aquatic Toxicity"

## Environment
The most important python packages are:

python == 3.9.25

pytorch == 2.1.0+cu118

dgl == 2.2.1+cu118

To replicate or devleop models more conveniently, the environment file <environmental.yml> is provided to install environment directly.
```bash
conda env create -f environment.yml
```
## Main

### Data
`AuqaTox_scr.csv`: Aquatic toxicity dataset with 15976 SMILES codes and discrete labels, which is involved 6 aquatic toxicity end points: fish acute tocicity(FishAT), fish chronic toxicity (FishCT), invetebrates acute toxicity (DMAT), inveterbrates chronic toxicity (DMCT), algal acute toxicity (AlgAT), algal chronic toxicity (AlgCT).

`Chemical inventories`: This dataset has approximately 1000000 compounds, which can be acquired at: https://doi.org/10.1021/acs.est.3c03860

### MTL-GCN
MTL-GCN model codes consist of folder'`utils`', file '`build_graph_dataset.py`' and '`Toxicity_MGA_MT.py`'.The `utils` file contains the codes related to molecular graph encoding and the model architecture.
step1: run `build_graph_dataset.py` to create molecular graph. `note: modify the file path and name of the training data as needed

step2: 





























