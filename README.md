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
`Chemical 
