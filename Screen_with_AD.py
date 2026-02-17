#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

import torch
from rdkit import Chem
import dgl
import warnings
warnings.filterwarnings('ignore')
# Import modules
from utils.MY_GNN import MGA

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using device: {device}")
from metAppDomain_ADM import NSG
HAS_ADSAL = True


# =============================================================================
# Configuration Parameters - Modify your settings here
# =============================================================================

# Prediction mode selection: 'file', 'single'
PREDICTION_MODE = 'file'  # 

# Input/Output file configuration
INPUT_FILE = '/.csv'        # Input compound file
OUTPUT_FILE = '/.xlsx'  # Output file name

# Single SMILES prediction parameters (used when PREDICTION_MODE='single')
SINGLE_SMILES = 'Cc1cccc2ccccc12'  # SMILES to predict

# Optimal applicability domain parameters for each endpoint (different thresholds for different endpoints)
TASK_SPECIFIC_AD_PARAMS = {
    'FishAT': {
        'OPTIMAL_DENSLB': 0.30,
        'OPTIMAL_LDUB': 0.40,
        'OPTIMAL_A':8
    },
    'FishCT': {
        'OPTIMAL_DENSLB': 0.10,
        'OPTIMAL_LDUB': 0.90,
        'OPTIMAL_A':5
    },
    'CruCT': {
        'OPTIMAL_DENSLB': 0.10,
        'OPTIMAL_LDUB': 0.90,
        'OPTIMAL_A':8
    },
    'CruAT': {
        'OPTIMAL_DENSLB': 0.10,
        'OPTIMAL_LDUB': 0.90,
        'OPTIMAL_A':7
    },
    'AlgAT': {
        'OPTIMAL_DENSLB': 0.10,
        'OPTIMAL_LDUB': 0.330,
        'OPTIMAL_A': 9
    },
    'AlgCT': {
        'OPTIMAL_DENSLB': 0.50,
        'OPTIMAL_LDUB': 0.80,
        'OPTIMAL_A':8
    }
}

# =============================================================================
# Custom molecular fingerprints for each task (new interface)
# =============================================================================
# Available fingerprint types (depends on NSG class support):
# 1. 'MACCS_keys': 167-bit structural keys (commonly used, fast computation)
# 2. 'PubChem': 881-bit PubChem fingerprint (requires scikit-fingerprints library)
# 3. 'Avalon': 512-bit Avalon fingerprint (requires RDKit Avalon support)

TASK_SPECIFIC_FP_PARAMS = {
    'FishAT': {
        'type': 'PubChem',   # Fingerprint type
    },
    'FishCT': {
        'type': 'MACCS_keys',      # Use MACCS_keys
    },
    'CruCT': {
        'type': 'PubChem',      # Use PubChem
    },
    'CruAT': {
        'type': 'MACCS_keys'       # Use MACCS_keys
    },
    'AlgAT': {
        'type': 'MACCS_keys',      # Use MACCS_keys
    },
    'AlgCT': {
        'type': 'MACCS_keys',      # Use MACCS_keys
    }
}

# Model and data paths (consistent with Toxicity_MGA_MT.py)
MODEL_PATH = '/.pth'  # Model path consistent with task_name
TRAINING_DATA_PATH = '/.csv'  # Training data path

# Batch processing parameters (consistent with Toxicity_MGA_MT.py)
BATCH_SIZE = 256  # Batch size for model prediction
AD_BATCH_SIZE = 10000  # Batch size for applicability domain calculation (to avoid memory overflow)
# MTL-scr model parameters (must be exactly consistent with Toxicity_MGA_MT.py training)
MODEL_ARGS = {
    'in_feats': 40,
    'rgcn_hidden_feats': [256, 128],
    'n_tasks': 6,
    'classifier_hidden_feats': 128,
    'rgcn_drop_out': 0.4,  # Value used during training
    'dropout': 0.3,        # Value used during training
    'loop': True
}

# Task name list (consistent with training - must be exactly the same order as select_task_list in Toxicity_MGA_MT.py)
TASK_NAMES = ['FishCT','CruAT', 'FishAT',  'CruCT', 'AlgAT','AlgCT']  # Order must be consistent with training!

# Graph construction parameters
GRAPH_ARGS = {
    'atom_data_field': 'atom',
    'bond_data_field': 'etype'
}

# =============================================================================
# Graph Construction and Prediction Functions
# =============================================================================

def construct_molecule_graph(smiles):
    """Construct molecular graph - exactly consistent with training"""
    from utils.build_dataset import construct_RGCN_bigraph_from_smiles

    try:
        # Use the same graph construction function as during training
        g = construct_RGCN_bigraph_from_smiles(smiles)
        return g
    except Exception as e:
        raise ValueError(f"Failed to construct molecular graph - SMILES: {smiles}, Error: {str(e)}")

class MTLScrPredictor:
    """Multi-task aquatic toxicity predictor based on MGA model"""

    def __init__(self, model_path, model_args=None):
        """Initialize predictor"""
        self.model_path = model_path
        self.device = device
        self.model_args = model_args or MODEL_ARGS
        self.task_names = TASK_NAMES

        self._load_model()

    def _load_model(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        print(f" Loading MTL-scr model: {self.model_path}")
        print(f" Using device: {self.device}")

        # Build model - use parameter names consistent with Toxicity_MGA_MT.py
        self.model = MGA(
            in_feats=self.model_args['in_feats'],
            rgcn_hidden_feats=self.model_args['rgcn_hidden_feats'],
            n_tasks=self.model_args['n_tasks'],
            classifier_hidden_feats=self.model_args['classifier_hidden_feats'],
            rgcn_drop_out=self.model_args['rgcn_drop_out'],
            dropout=self.model_args['dropout'],  # Note: MGA model parameter name is dropout
            loop=self.model_args['loop']
        )

        # Load model weights - compatible with different save formats
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Check checkpoint format
        if 'model_state_dict' in checkpoint:
            # New format: contains complete training state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f" Loaded complete checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            # Old format: directly model state dict
            self.model.load_state_dict(checkpoint)
            print(f" Loaded model state dict")

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print(f" MTL-scr model loaded successfully (device: {self.device})")

    def _validate_smiles(self, smiles):
        """Validate SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES string"

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return True, canonical_smiles
        except Exception as e:
            return False, f"SMILES processing error: {str(e)}"

    def predict_single(self, smiles):
        """Predict multi-task toxicity for a single SMILES"""
        is_valid, result = self._validate_smiles(smiles)
        if not is_valid:
            return {'smiles': smiles, 'error': result}

        canonical_smiles = result

        try:
            # Construct molecular graph
            g = construct_molecule_graph(canonical_smiles)

            # Batch graph
            bg = dgl.batch([g]).to(self.device)
            atom_feats = bg.ndata[GRAPH_ARGS['atom_data_field']].float().to(self.device)  # Ensure float type
            bond_feats = bg.edata[GRAPH_ARGS['bond_data_field']].long().to(self.device)   # Ensure long type

            with torch.no_grad():
                # Get multi-task prediction results
                predictions = self.model(bg, atom_feats, bond_feats)
                predictions = torch.sigmoid(predictions).cpu().numpy()[0]  # Apply sigmoid and convert to numpy

            # Build result dictionary
            result_dict = {
                'smiles': smiles,
                'canonical_smiles': canonical_smiles,
                'error': None
            }

            # Add prediction results for each task
            for i, task_name in enumerate(self.task_names):
                probability = float(predictions[i])
                prediction = int(probability > 0.5)
                result_dict[f'{task_name}_prediction'] = prediction
                result_dict[f'{task_name}_probability'] = probability
                result_dict[f'{task_name}_label'] = 'Toxic' if prediction == 1 else 'Non-toxic'

            return result_dict

        except Exception as e:
            return {'smiles': smiles, 'error': f"Prediction error: {str(e)}"}

def predict_on_input_file(input_file_path, model_path=None, output_file_path=None):
    """
    Perform MTL-scr multi-task prediction on input file

    Args:
        input_file_path: Input file path (Excel format)
        model_path: Model file path, use default model if None
        output_file_path: original_file_predicted
    """
    print("Starting MTL-scr multi-task prediction on input file...")

    # 1. Check input file
    if not os.path.exists(input_file_path):
        print(f"Input file does not exist: {input_file_path}")
        return

    # 2. Determine model path
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        return

    # 3. Determine output file path
    if output_file_path is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_predicted.xlsx"

    print(f"Input file: {input_file_path}")
    print(f"Model file: {model_path}")
    print(f"Output file: {output_file_path}")
    print(f"Using device: {device}")

    # 4. Load MTL-scr model
    try:
        # Build model - use parameter names consistent with Toxicity_MGA_MT.py
        model = MGA(
            in_feats=MODEL_ARGS['in_feats'],
            rgcn_hidden_feats=MODEL_ARGS['rgcn_hidden_feats'],
            n_tasks=MODEL_ARGS['n_tasks'],
            classifier_hidden_feats=MODEL_ARGS['classifier_hidden_feats'],
            rgcn_drop_out=MODEL_ARGS['rgcn_drop_out'],
            dropout=MODEL_ARGS['dropout'],  # Note: MGA model parameter name is dropout
            loop=MODEL_ARGS['loop']
        )

        # Load model weights - compatible with different save formats
        checkpoint = torch.load(model_path, map_location=device)

        # Check checkpoint format
        if 'model_state_dict' in checkpoint:
            # New format: contains complete training state
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded complete checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            # Old format: directly model state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model state dict")

        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()

        print(f"MTL-scr model loaded successfully (device: {device})")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return

    # 5. Load input data
    try:
        # Read Excel file, ensure Canonical SMILES column is string type
        df = pd.read_csv(input_file_path, dtype={'Canonical SMILES': str})
        print(f"Number of samples: {len(df)}")

        # Check required columns
        required_columns = ['Canonical SMILES']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Input data missing required columns: {missing_columns}")
            return

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return

    # 6. Prepare result DataFrame - keep only necessary information
    result_df = df[['Canonical SMILES']].copy()  # Keep only original smiles
    result_df['error'] = ''

    # Add simplified prediction columns for each task
    for task_name in TASK_NAMES:
        result_df[f'{task_name}_prediction'] = -1      # Prediction value 0/1
        result_df[f'{task_name}_probability'] = 0.0    # Prediction probability
        result_df[f'{task_name}_in_AD'] = False        # Whether in applicability domain

    # 7. Validate SMILES and perform prediction
    print("Starting multi-task prediction...")
    valid_indices = []
    valid_canonical_smiles = []

    for idx, row in df.iterrows():
        smiles = row['Canonical SMILES']

        # Check if SMILES is a valid string
        if pd.isna(smiles):
            result_df.loc[idx, 'error'] = "SMILES is empty"
            continue

        # Ensure SMILES is string type
        smiles = str(smiles).strip()

        if smiles == '' or smiles == 'nan':
            result_df.loc[idx, 'error'] = "SMILES is empty"
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result_df.loc[idx, 'error'] = "Invalid SMILES string"
                continue

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            valid_indices.append(idx)
            valid_canonical_smiles.append(canonical_smiles)

        except Exception as e:
            result_df.loc[idx, 'error'] = f"SMILES processing error: {str(e)}"

    print(f"Valid samples: {len(valid_indices)}/{len(df)}")

    if len(valid_indices) == 0:
        print("No valid samples to predict")
        return

    # 8. Batch prediction
    try:
        batch_size = BATCH_SIZE
        batch_list = [valid_indices[i:i+batch_size] for i in range(0, len(valid_indices), batch_size)]

        for batch_idx, batch_indices in enumerate(batch_list):
            print(f"Processing batch {batch_idx + 1}/{len(batch_list)}")

            batch_smiles = [valid_canonical_smiles[valid_indices.index(idx)] for idx in batch_indices]

            # Construct molecular graphs
            graphs = []
            for smi in batch_smiles:
                try:
                    graphs.append(construct_molecule_graph(smi))
                except Exception as e:
                    print(f"Graph construction failed: {smi}, {str(e)}")
                    graphs.append(None)

            # Filter valid graphs
            valid_graphs = [g for g in graphs if g is not None]
            if not valid_graphs:
                print(f"Batch {batch_idx + 1} has no valid molecular graphs")
                continue

            # Batch graphs
            bg = dgl.batch(valid_graphs).to(device)
            atom_feats = bg.ndata[GRAPH_ARGS['atom_data_field']].float().to(device)  # Ensure float type
            bond_feats = bg.edata[GRAPH_ARGS['bond_data_field']].long().to(device)   # Ensure long type

            # Predict
            with torch.no_grad():
                predictions = model(bg, atom_feats, bond_feats)
                predictions = torch.sigmoid(predictions).cpu().numpy()  # Apply sigmoid

                # Save prediction results - simplified version
                valid_idx = 0
                for i, orig_idx in enumerate(batch_indices):
                    if graphs[i] is not None:  # Only process results for valid graphs
                        for j, task_name in enumerate(TASK_NAMES):
                            probability = float(predictions[valid_idx, j])
                            prediction = int(probability > 0.5)

                            result_df.loc[orig_idx, f'{task_name}_prediction'] = prediction
                            result_df.loc[orig_idx, f'{task_name}_probability'] = probability
                        valid_idx += 1
                    else:
                        # Set error message for invalid graphs
                        result_df.loc[orig_idx, 'error'] = "Molecular graph construction failed"

                # Clean GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("Multi-task prediction completed")

    except Exception as e:
        print(f"Prediction process error: {str(e)}")
        # Mark error for all valid indices
        for idx in valid_indices:
            result_df.loc[idx, 'error'] = f"Prediction error: {str(e)}"

    # 9. Save results
    try:
        # Save as Excel to prevent special character issues
        result_df.to_excel(output_file_path, index=False)
        print(f"Results saved to: {output_file_path}")

        # Display simplified statistics
        valid_predictions = result_df[result_df['error'] == '']
        errors = result_df[result_df['error'] != '']

        print(f"\nPrediction statistics:")
        print(f"Successful predictions: {len(valid_predictions)}/{len(result_df)}")
        if len(errors) > 0:
            print(f"Failed samples: {len(errors)}")

        if len(valid_predictions) > 0:
            # Simplified task statistics
            for task_name in TASK_NAMES:
                pred_col = f'{task_name}_prediction'
                if pred_col in valid_predictions.columns:
                    toxic_count = (valid_predictions[pred_col] == 1).sum()
                    print(f"{task_name}: {toxic_count}/{len(valid_predictions)} toxic")

    except Exception as e:
        print(f"Failed to save results: {str(e)}")

# Exponential weight function
def expWt(x, a=15, eps=1e-6):
    """Exponential weight function"""
    return np.exp(-a*(1-x)/(x + eps))

# =============================================================================
# Core Functions
# =============================================================================

def load_training_data_for_task(file_path, task_name):
    """Load training data for a specific task"""
    print(f"Loading training data for task {task_name}: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file does not exist: {file_path}")

    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ['smiles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Training set missing required columns: {missing_cols}")

    # Create labels for specific task
    if task_name in df.columns:
        df['y'] = df[task_name].fillna(0).astype(int)
        # Keep only samples with valid data for this task
        valid_samples = df[df[task_name].notna()]
        print(f"Task {task_name} valid training samples: {len(valid_samples)}")

        # Check if there are enough training samples
        if len(valid_samples) == 0:
            raise ValueError(f"Task {task_name} has no valid training samples (all values are NaN)")

        # Check if SMILES column has null values
        null_smiles = valid_samples['smiles'].isna().sum()
        if null_smiles > 0:
            print(f"  Warning: Found {null_smiles} empty SMILES, will be filtered")
            valid_samples = valid_samples[valid_samples['smiles'].notna()]
            print(f"  Valid samples after filtering: {len(valid_samples)}")
    else:
        raise ValueError(f"Task column not found in training set: {task_name}")

    # Prepare data
    df_clean = valid_samples[['smiles', 'y']].copy()
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean

def run_prediction(input_file, model_path, temp_output):
    """Run toxicity prediction"""
    print("Step 1: Run toxicity prediction")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # Run prediction
    predict_on_input_file(input_file, model_path, temp_output)

    if not os.path.exists(temp_output):
        raise RuntimeError("Prediction failed, prediction file not generated")

    print("Prediction completed")
    return temp_output

# Modified: added weight_a parameter
def calculate_ad_metrics_for_task(df_train, df_query, task_name, batch_size=10000, fp_settings=None, weight_a=15):
    """Calculate applicability domain metrics for a specific task (batch processing to avoid memory overflow)
    
    Args:
        df_train: Training data
        df_query: Query data
        task_name: Task name
        batch_size: Number of samples per batch, default 10000
        fp_settings: (new) Fingerprint settings dict, containing 'type' and other parameters
        weight_a: (new) Parameter a for exponential weight function
    """
    print(f"Step 2: Calculate applicability domain metrics for task {task_name}")
    
    # Set default fingerprint
    if fp_settings is None:
        fp_settings = {'type': 'MACCS_keys'}
    
    fp_type = fp_settings.get('type', 'MACCS_keys')
    # Extract other parameters besides 'type' (such as radius, nBits) as kwargs
    fp_kwargs = {k: v for k, v in fp_settings.items() if k != 'type'}

    try:
        # Check training data
        if len(df_train) == 0:
            raise ValueError(f"Task {task_name} has no valid training data")

        print(f"  Training set samples: {len(df_train)}")
        print(f"  Query set samples: {len(df_query)}")

        # Create NSG object
        nsg = NSG(df_train, yCol='y', smiCol='smiles')        # Calculate molecular fingerprint similarity (dynamic call)
        print(f"  Calculating molecular fingerprint similarity: using {fp_type} {fp_kwargs if fp_kwargs else ''}...")
        try:
            # Dynamically pass fingerprint type and parameters here
            nsg.calcPairwiseSimilarityWithFp(fp_type, **fp_kwargs)
        except Exception as e:
            print(f"  Fingerprint {fp_type} calculation failed, falling back to MACCS_keys. Error: {e}")
            nsg.calcPairwiseSimilarityWithFp('MACCS_keys')

        # Extract only SMILES column for applicability domain calculation, ensure correct data type
        df_query_smiles = df_query[['Canonical SMILES']].copy()
        
        # Save original index
        original_index = df_query_smiles.index
        
        # Ensure SMILES column is string type
        df_query_smiles['Canonical SMILES'] = df_query_smiles['Canonical SMILES'].astype(str)
        
        # Reset index to avoid index mismatch issues
        df_query_smiles_reset = df_query_smiles.reset_index(drop=True)
        
        # Batch processing to avoid memory overflow
        total_samples = len(df_query_smiles_reset)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"  Batch processing: total samples {total_samples}, batch size {batch_size}, total batches {num_batches}")
        
        all_ad_metrics = []
        
        # Define weight parameter dictionary
        weight_params = {'a': weight_a}
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            # Only show detailed log for first batch to avoid flooding
            if batch_idx == 0 or batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})...")
            
            # Get current batch data
            df_batch = df_query_smiles_reset.iloc[start_idx:end_idx]
            
            # Generate query-training similarity matrix
            dfQTSM_batch = nsg.genQTSM(df_batch, 'Canonical SMILES')
            
            # Calculate applicability domain metrics (using exp weight function)
            # Modified: pass correct weight_params
            ad_metrics_batch = nsg.queryADMetrics(
                dfQTSM_batch, 
                wtFunc1=expWt,
                kw1=weight_params, 
                wtFunc2=expWt,
                kw2=weight_params,
                code='|exp'
            )
            
            all_ad_metrics.append(ad_metrics_batch)
            
            # Clean memory
            del dfQTSM_batch
            del ad_metrics_batch
            
        # Merge results from all batches
        print(f"  Merging results from all batches...")
        ad_metrics = pd.concat(all_ad_metrics, axis=0, ignore_index=True)
        
        # Add task prefix to metrics
        ad_metrics_renamed = {}
        for col in ad_metrics.columns:
            ad_metrics_renamed[f'{task_name}_{col}'] = ad_metrics[col]
            
        ad_metrics_df = pd.DataFrame(ad_metrics_renamed)
          # Restore original index
        ad_metrics_df.index = original_index
        
        print(f"Task {task_name} applicability domain metrics calculation completed")
        return ad_metrics_df
        
    except Exception as e:
        print(f"  Error calculating applicability domain metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def apply_multi_task_ad_criteria(df_query):
    """Apply applicability domain criteria for multi-task (using task-specific thresholds)"""
    print("Step 3: Multi-task applicability domain assessment (using task-specific thresholds)")

    if not HAS_ADSAL:
        print("Warning: Applicability domain assessment function unavailable, skipping AD analysis")
        # Add default applicability domain information for each task
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_in_applicability_domain'] = True
            df_query[f'{task_name}_ad_reason'] = "Applicability domain assessment function unavailable"
        return df_query

    df_result = df_query.copy()

    # Calculate applicability domain for each task
    for task_name in TASK_NAMES:
        # Get task-specific thresholds
        if task_name not in TASK_SPECIFIC_AD_PARAMS:
            raise ValueError(f"Task {task_name} has no defined applicability domain parameters")

        task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
        task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']

        print(f"\nProcessing task: {task_name}")
        print(f"  Similarity density threshold (densLB): {task_densLB}")
        print(f"  Local discontinuity threshold (LdUB): {task_LdUB}")

        try:
            # Load training data for this task
            df_train_task = load_training_data_for_task(TRAINING_DATA_PATH, task_name)

            # Calculate applicability domain metrics for this task (using batch processing)
            # Note: custom fp and weight_a not used here, this is old interface,
            # can be added if needed, but this script mainly uses apply_simplified_multi_task_ad_criteria
            ad_metrics_task = calculate_ad_metrics_for_task(df_train_task, df_query, task_name, batch_size=AD_BATCH_SIZE)

            # Merge applicability domain metrics
            for col in ad_metrics_task.columns:
                df_result[col] = ad_metrics_task[col]

            # Applicability domain criteria
            density_col = f'{task_name}_simiDensity|exp'
            ld_col = f'{task_name}_simiWtLD_w|exp'

            if density_col in df_result.columns and ld_col in df_result.columns:
                ad_condition = (
                    (df_result[density_col] >= task_densLB) &
                    (df_result[ld_col] <= task_LdUB)
                )

                df_result[f'{task_name}_in_applicability_domain'] = ad_condition
                df_result[f'{task_name}_ad_density_value'] = df_result[density_col]
                df_result[f'{task_name}_ad_ld_value'] = df_result[ld_col]
                df_result[f'{task_name}_ad_densLB_threshold'] = task_densLB
                df_result[f'{task_name}_ad_LdUB_threshold'] = task_LdUB

                # Add applicability domain assessment reason
                def get_task_ad_reason(row, task, dens_threshold, ld_threshold):
                    density_val = row[f'{task}_ad_density_value']
                    ld_val = row[f'{task}_ad_ld_value']

                    if row[f'{task}_in_applicability_domain']:
                        return f"{task}: Within applicability domain"
                    else:
                        reasons = []
                        if density_val < dens_threshold:
                            reasons.append(f"Similarity density({density_val:.3f}) < threshold({dens_threshold})")
                        if ld_val > ld_threshold:
                            reasons.append(f"Local discontinuity({ld_val:.3f}) > threshold({ld_threshold})")
                        return f"{task}: " + "; ".join(reasons)

                df_result[f'{task_name}_ad_reason'] = df_result.apply(
                    lambda row: get_task_ad_reason(row, task_name, task_densLB, task_LdUB), axis=1
                )                # Statistics for this task's applicability domain results
                total_compounds = len(df_result)
                in_domain_count = df_result[f'{task_name}_in_applicability_domain'].sum()
                print(f"  {task_name} within AD: {in_domain_count}/{total_compounds} ({in_domain_count/total_compounds*100:.1f}%)")

        except Exception as e:
            print(f"  Warning: Task {task_name} applicability domain calculation failed: {str(e)}")
            # Set default values
            df_result[f'{task_name}_in_applicability_domain'] = False
            df_result[f'{task_name}_ad_reason'] = f"Applicability domain calculation failed: {str(e)}"

    return df_result

def apply_simplified_multi_task_ad_criteria(df_query):
    """Simplified multi-task applicability domain assessment - keep only necessary information (using task-specific thresholds and fingerprints)"""
    print("Step 3: Simplified multi-task applicability domain assessment (using task-specific thresholds and fingerprints)")
    
    if not HAS_ADSAL:
        print("Warning: Applicability domain assessment function unavailable, skipping AD analysis")
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_in_AD'] = True
        return df_query
        
    df_result = df_query.copy()
    
    # Check if error column exists (consistent with original code logic)
    if 'error' not in df_result.columns:
        print("Warning: No error column in data, assuming all samples are valid")
        df_valid = df_result.copy()
        valid_mask = pd.Series([True] * len(df_result), index=df_result.index)
    else:
        df_result['error'] = df_result['error'].fillna('')
        valid_mask = df_result['error'] == ''
        df_valid = df_result[valid_mask].copy()
        print(f"Total samples: {len(df_result)}, Valid samples: {len(df_valid)}")

    if len(df_valid) == 0:
        print("Warning: No valid samples for applicability domain calculation")
        return df_result

    # Calculate applicability domain for each task
    for task_name in TASK_NAMES:
        # 1. Get task-specific AD thresholds
        if task_name not in TASK_SPECIFIC_AD_PARAMS:
            raise ValueError(f"Task {task_name} has no defined applicability domain parameters")
            
        task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
        task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']
        task_a = TASK_SPECIFIC_AD_PARAMS[task_name].get('OPTIMAL_A', 15)
        
        # 2. Get task-specific fingerprint settings (new)
        fp_settings = TASK_SPECIFIC_FP_PARAMS.get(task_name, {'type': 'MACCS_keys'})

        print(f"\nProcessing task: {task_name}")
        print(f"  Using fingerprint: {fp_settings.get('type')} { {k:v for k,v in fp_settings.items() if k!='type'} }")
        print(f"  Weight parameter a: {task_a}") # Print confirmation
        print(f"  Similarity density threshold (densLB): {task_densLB}")
        print(f"  Local discontinuity threshold (LdUB): {task_LdUB}")
        
        try:
            # Load training data for this task
            df_train_task = load_training_data_for_task(TRAINING_DATA_PATH, task_name)
            
            # Calculate applicability domain metrics for this task (pass fp_settings and weight_a)
            ad_metrics_task = calculate_ad_metrics_for_task(
                df_train_task, 
                df_valid, 
                task_name, 
                batch_size=AD_BATCH_SIZE,
                fp_settings=fp_settings,
                weight_a=task_a   # Previously errored here, now fixed
            )

            # Applicability domain criteria (logic unchanged)
            density_col = f'{task_name}_simiDensity|exp'
            ld_col = f'{task_name}_simiWtLD_w|exp'
            
            if density_col in ad_metrics_task.columns and ld_col in ad_metrics_task.columns:
                ad_condition = (
                    (ad_metrics_task[density_col] >= task_densLB) & 
                    (ad_metrics_task[ld_col] <= task_LdUB)
                )
                
                # Merge applicability domain results back to original DataFrame
                df_result.loc[valid_mask, f'{task_name}_in_AD'] = ad_condition.values
                  # Statistics
                total_compounds = len(df_valid)
                in_domain_count = ad_condition.sum()
                print(f"  {task_name} within AD: {in_domain_count}/{total_compounds} ({in_domain_count/total_compounds*100:.1f}%)")
                
        except Exception as e:
            print(f"  Warning: Task {task_name} applicability domain calculation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            df_result.loc[valid_mask, f'{task_name}_in_AD'] = False
            
    return df_result

def generate_summary(df_result):
    """Generate simplified multi-task result summary"""
    print("\nPrediction Result Summary:")
    print("=" * 60)

    # Only show detailed information for first 5 samples
    display_count = min(5, len(df_result))

    for idx in range(display_count):
        row = df_result.iloc[idx]
        print(f"\nSample {idx+1}:")
        print(f"  SMILES: {row['Canonical SMILES']}")

        if row.get('error', ''):
            print(f"  Error: {row['error']}")
            continue

        print(f"  Prediction Results:")
        for task_name in TASK_NAMES:
            pred = row.get(f'{task_name}_prediction', -1)
            prob = row.get(f'{task_name}_probability', 0)
            in_ad = row.get(f'{task_name}_in_AD', False)

            if pred != -1:
                status = "Toxic" if pred == 1 else "Non-toxic"
                ad_status = "In AD" if in_ad else "Out of AD"
                print(f"    {task_name}: {status} ({prob:.3f}) - {ad_status}")

    if len(df_result) > display_count:
        print(f"\n... {len(df_result) - display_count} more samples")

    # Overall statistics
    valid_samples = df_result[df_result['error'] == '']
    if len(valid_samples) > 0:
        print(f"\nOverall Statistics:")
        for task_name in TASK_NAMES:
            pred_col = f'{task_name}_prediction'
            ad_col = f'{task_name}_in_AD'

            if pred_col in valid_samples.columns:
                toxic_count = (valid_samples[pred_col] == 1).sum()
                if ad_col in valid_samples.columns:
                    in_ad_count = valid_samples[ad_col].sum()
                    print(f"  {task_name}: {toxic_count}/{len(valid_samples)} toxic, {in_ad_count}/{len(valid_samples)} in AD")
                else:
                    print(f"  {task_name}: {toxic_count}/{len(valid_samples)} toxic")

    print("=" * 60)

def run_single_prediction_with_ad():
    """Run single SMILES prediction with applicability domain assessment"""
    print("=" * 60)
    print("Single SMILES Multi-task Prediction with Applicability Domain Assessment")
    print("=" * 60)

    print(f"SMILES: {SINGLE_SMILES}")
    print("Using task-specific applicability domain thresholds")
    print()

    try:
        # Step 1: Multi-task toxicity prediction
        predictor = MTLScrPredictor(MODEL_PATH)
        prediction_result = predictor.predict_single(SINGLE_SMILES)

        if prediction_result.get('error'):
            print(f"Prediction failed: {prediction_result['error']}")
            return

        print("Step 1: Multi-task toxicity prediction completed")
        for task_name in TASK_NAMES:
            pred_label = prediction_result.get(f'{task_name}_label', 'N/A')
            pred_prob = prediction_result.get(f'{task_name}_probability', 0)
            print(f"  {task_name}: {pred_label} (probability: {pred_prob:.4f})")

        # Step 2: Create query data
        query_data = {
            'Canonical SMILES': [prediction_result['canonical_smiles']],
            'compound_name': [f"Query_compound_{SINGLE_SMILES[:10]}"]
        }
        df_query = pd.DataFrame(query_data)

        # Add multi-task prediction results to query data
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_prediction'] = prediction_result.get(f'{task_name}_prediction', -1)
            df_query[f'{task_name}_probability'] = prediction_result.get(f'{task_name}_probability', 0.0)
            df_query[f'{task_name}_label'] = prediction_result.get(f'{task_name}_label', 'N/A')

        # Step 3: Multi-task applicability domain assessment (using task-specific thresholds)
        df_final = apply_multi_task_ad_criteria(df_query)

        # Display results
        print("\n" + "=" * 60)
        print("Complete Multi-task Analysis Results:")
        print("=" * 60)

        row = df_final.iloc[0]

        print(f"SMILES: {SINGLE_SMILES}")
        print(f"Canonical SMILES: {row['Canonical SMILES']}")

        print(f"\nMulti-task Toxicity Prediction:")
        for task_name in TASK_NAMES:
            pred_label = row.get(f'{task_name}_label', 'N/A')
            pred_prob = row.get(f'{task_name}_probability', 0)
            in_ad = row.get(f'{task_name}_in_applicability_domain', False)
            ad_status = "Within AD" if in_ad else "Outside AD"

            # Get task thresholds
            if task_name not in TASK_SPECIFIC_AD_PARAMS:
                raise ValueError(f"Task {task_name} has no defined applicability domain parameters")

            task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
            task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']

            density_val = row.get(f'{task_name}_ad_density_value', 0)
            ld_val = row.get(f'{task_name}_ad_ld_value', 0)

            print(f"  {task_name}: {pred_label} (probability: {pred_prob:.4f}) - {ad_status}")
            print(f"    Similarity density: {density_val:.4f} (threshold: {task_densLB})")
            print(f"    Local discontinuity: {ld_val:.4f} (threshold: {task_LdUB})")

        # Provide recommendations
        print(f"\nRecommendations:")
        for task_name in TASK_NAMES:
            in_ad = row.get(f'{task_name}_in_applicability_domain', False)
            if in_ad:
                suggestion = f"  {task_name}: Prediction result is reliable, recommended for use"
            else:
                suggestion = f"  {task_name}: Prediction reliability is low, experimental verification recommended"
            print(suggestion)

        # Save results
        output_file = f"single_prediction_result_{SINGLE_SMILES[:10].replace('/', '_')}.xlsx"
        df_final.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_prediction_pipeline():
    """Test basic functionality of the prediction pipeline"""
    print("=" * 60)
    print("Testing Prediction Pipeline")
    print("=" * 60)

    # Test SMILES list
    test_smiles = [
        'CCO',  # Ethanol - simple molecule
        'CC(C)O',  # Isopropanol
        'c1ccccc1',  # Benzene        'CCN(CC)CC',  # Triethylamine
        'invalid_smiles'  # Invalid SMILES for error handling test
    ]

    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model file does not exist: {MODEL_PATH}")
            print("Please ensure model file path is correct")
            return False

        print(f"[OK] Model file exists: {MODEL_PATH}")

        # Initialize predictor
        print("Initializing predictor...")
        predictor = MTLScrPredictor(MODEL_PATH)
        print("[OK] Predictor initialized successfully")        # Test single SMILES prediction
        print("\nTesting single SMILES prediction:")
        for i, smiles in enumerate(test_smiles):
            print(f"\nTest {i+1}: {smiles}")
            result = predictor.predict_single(smiles)

            if result.get('error'):
                print(f"  [ERROR] Prediction failed: {result['error']}")
            else:
                print(f"  [OK] Prediction successful")
                print(f"  Canonical SMILES: {result['canonical_smiles']}")
                for task_name in TASK_NAMES:
                    pred_label = result.get(f'{task_name}_label', 'N/A')
                    pred_prob = result.get(f'{task_name}_probability', 0)
                    print(f"    {task_name}: {pred_label} (probability: {pred_prob:.4f})")

        print("\n[OK] Prediction pipeline test completed")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("MTL-scr Multi-task Chemical Toxicity Prediction with Applicability Domain Assessment Integrated Analysis")
    print("=" * 60)

    print(f"Prediction mode: {PREDICTION_MODE}")
    print(f"Model: MTL-scr (Multi-task Learning)")
    print(f"Tasks: {', '.join(TASK_NAMES)}")
    if PREDICTION_MODE == 'file':
        print(f"Input file: {INPUT_FILE}")
        print(f"Output file: {OUTPUT_FILE}")
    elif PREDICTION_MODE == 'single':
        print(f"Single SMILES: {SINGLE_SMILES}")
    elif PREDICTION_MODE == 'test':
        print(f"Test mode: Validate prediction pipeline")
    print(f"Using task-specific applicability domain thresholds")
    print()
    
    try:
        # Check if applicability domain assessment function is available
        if not HAS_ADSAL:
            print("Warning: Applicability domain assessment function unavailable, only toxicity prediction will be performed")

        if PREDICTION_MODE == 'test':
            # Test mode
            print("Running prediction pipeline test...")
            success = test_prediction_pipeline()
            if success:
                print("\n[OK] Prediction pipeline test passed! Ready for formal prediction.")
            else:
                print("\n[ERROR] Prediction pipeline test failed! Please check configuration.")

        elif PREDICTION_MODE == 'single':
            if HAS_ADSAL:
                run_single_prediction_with_ad()
            else:
                # Only perform toxicity prediction
                predictor = MTLScrPredictor(MODEL_PATH)
                prediction_result = predictor.predict_single(SINGLE_SMILES)

                if prediction_result.get('error'):
                    print(f"Prediction failed: {prediction_result['error']}")
                    return

                print("Multi-task toxicity prediction results:")
                for task_name in TASK_NAMES:
                    pred_label = prediction_result.get(f'{task_name}_label', 'N/A')
                    pred_prob = prediction_result.get(f'{task_name}_probability', 0)
                    print(f"  {task_name}: {pred_label} (probability: {pred_prob:.4f})")

        elif PREDICTION_MODE == 'file':
            # Check input file
            if not os.path.exists(INPUT_FILE):
                raise FileNotFoundError(f"Input file does not exist: {INPUT_FILE}")

            # Step 1: Run toxicity prediction
            temp_prediction_file = 'temp_prediction.xlsx'
            run_prediction(INPUT_FILE, MODEL_PATH, temp_prediction_file)

            if HAS_ADSAL:
                # Step 2&3: Multi-task applicability domain assessment (using task-specific thresholds)
                # Note: using read_excel to read temp_prediction_file here
                df_prediction = pd.read_excel(temp_prediction_file)
                print(f"\nRead prediction file: {temp_prediction_file}")
                print(f"Columns: {df_prediction.columns.tolist()}")
                print(f"Data shape: {df_prediction.shape}")
                if 'error' in df_prediction.columns:
                    # Note: Empty strings read by pd.read_excel may be NaN, handle here for statistics
                    error_check = df_prediction['error'].fillna('')
                    error_count = (error_check != '').sum()
                    print(f"Samples with errors: {error_count}")

                df_final = apply_simplified_multi_task_ad_criteria(df_prediction)

                # Step 5: Save results
                df_final.to_excel(OUTPUT_FILE, index=False)
                print(f"\nResults saved to: {OUTPUT_FILE}")                # Step 6: Generate summary
                generate_summary(df_final)
            else:
                # Only save prediction results
                print(f"\nPrediction results saved to: {temp_prediction_file}")
                print("Warning: Applicability domain assessment not performed due to missing adsal module")

            # Clean up temporary files
            if HAS_ADSAL and os.path.exists(temp_prediction_file):
                os.remove(temp_prediction_file)

            print("\nAnalysis completed!")
        else:
            raise ValueError(f"Unsupported prediction mode: {PREDICTION_MODE}, supported modes: 'file', 'single', 'test'")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Clean up temporary files
        if os.path.exists('temp_prediction.xlsx'):
            os.remove('temp_prediction.xlsx')

        sys.exit(1)

if __name__ == "__main__":
    main()
