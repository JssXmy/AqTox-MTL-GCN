# MTL-GCN-AqTox
The files related to the paper published in Chemical Research in Toxicology:
"Multi-Task Graph Convolutional Network Model with Improved Performance and Attention Mechanisms for Identifying Chemicals with Aquatic Toxicity"

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
`AuqaTox_scr.csv`: Aquatic toxicity dataset with 15976 SMILES codes and discrete labels, which is involved 6 aquatic toxicity end points: fish acute tocicity(FishAT), fish chronic toxicity (FishCT), invetebrates acute toxicity (DMAT), inveterbrates chronic toxicity (DMCT), algal acute toxicity (AlgAT), algal chronic toxicity (AlgCT).

`Chemical inventories`: This dataset has approximately 1000000 compounds, which can be acquired from references: DOI: 10.1021/acs.est.3c03860)

### MTL-GCN
MTL-GCN model codes consist of folder'`utils`', file '`build_graph_dataset.py`' and '`Toxicity_MGA_MT.py`'.The `utils` file contains the codes related to molecular graph encoding and the model architecture.

Step1: Run `build_graph_dataset.py` to create molecular graph. `note: modify the file path and name of the training data as needed,the data example provided in the `AuqaTox_scr.csv` on data structures

Step2: Run the `Toxicity_MGA_MT.py` to train MTL-GCN model. Finally, you will receive the prediction results and performance of the model.

### Single-task (ST) Models
ST Models include ST-GCN and traditional machine learning (ML) models, involving RF, XGBoost, LightGBM.
The ST-GCN model shares the same algorithm as MTL-GCN. There is no need to generate separate molecular graph data for each endpoint. Once the multi-task data is created, simply update the '`args['select_task_list']`' in `Toxicity_MGA_ST.py` and ruin it then to build single-task models for different endpoints.

ST-ML Model codes related to ST traditional ML models is located in the '`ML_Modeling`' folder, which includes '`fp_generation_{fp type}.py`' for generating molecular fingerprints(fp) and '`{algorithms}_{fp type}_classification.py`'

### Applicability domain (AD)
Code related to the Applicability Domain is located in the '`AD`' folder. This includes data files '`TrainingSet.xlsx`' and '`ExternalSet_pred.xlsx`', as well as the code files '`AD.py`' and '`metAppDomain_ADM.py`'. (The code origninated from previous reference: DOI: 10.1021/acs.chemrestox.3c00074)

`metAppDomain_ADM.py`: Required files for structure activity landscape-based application domains (ADSAL);

`AD.py` : characterize the ADSAL of a model; Users can set different application domain stringency levels according to the instructions in the codes and their own needs, in order to achieve the function of improving the prediction of the MTL-GCN model.

`TrainingSet.xlsx` and `ExternalSet_pred.xlsx`: Training and e xternal validation sets used in the current study for developing the optimal MTL-GCN model. Note: The data structure in the training set and external validation sets provided here are not real but serve as examples only. The `AD.py` file can be successfully executed by following the structure of the example data.





























