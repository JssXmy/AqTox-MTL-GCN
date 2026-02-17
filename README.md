# MTL-GCN-AqTox
The files related to the paper:
"Multi-Task Graph Convolutional Network Model with Improved Performance and Broad Applicability Domains for Identifying Aquatic Toxic Chemicals"

## Environment
The most important python packages are:

python == 3.9.25

pytorch == 2.1.0+cu118

dgl == 2.2.1+cu118

To replicate or devleop models more conveniently, the environment file '`environmental.yml`' is provided to install environment directly.
```bash
conda env create -f environment.yml
```
## Main

### Data
`AuqaTox.csv`: Aquatic toxicity dataset with 17,514 SMILES codes and discrete labels, which is involved 6 aquatic toxicity endpoints: fish acute tocicity(FishAT), fish chronic toxicity (FishCT), crustacean acute toxicity (CruAT), crustacean chronic toxicity (CruCT), algal acute toxicity (AlgAT), algal chronic toxicity (AlgCT);

### Multi-task learning (MTL-GCN) Model
MTL-GCN framework codes consist of folder'`utils`', file '`build_graph_dataset.py`' and '`Toxicity_MTL_GCN.py`'.The `utils` file contains the codes related to molecular graph encoding and the model architecture.

Step1: Run `build_graph_dataset.py` to create molecular graphs. `note: modify the file path and name of the training data as needed,the data example provided in the `AuqaTox.csv` on data structures;

Step2: Run the `Toxicity_MTL_GCN.py` to train MTL-GCN model. Finally, you will receive the prediction results and performance of the model.

### Single-task learning (STL) Models
ST Models include ST-GCN and classical machine learning (ML) models, involving RF, XGBoost, LightGBM.
The ST-GCN model shares the same algorithm as MTL-GCN. There is no need to generate separate molecular graph data for each endpoint. Once the multi-task data is created, simply update the '`args['select_task_list']`' in `Toxicity_MGA_ST.py` and ruin it then to build single-task models for different endpoints;

ST-ML Model codes related to ST classical ML models is located in the '`ML_Modeling`' folder. Prior to execution, various molecular fingerprints must be calculated. The required input file structure and format are provided in this folder.






























