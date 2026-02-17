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
from utils.MY_GNN import MGA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from AD.metAppDomain_ADM import NSG
HAS_ADSAL = True

# 'file', 'single',
PREDICTION_MODE = 'file' 

INPUT_FILE = 'data/Chemcial_inventories.xlsx'        
OUTPUT_FILE = 'prediction/Chemcial_inventories_prediction_with_AD_results.xlsx' 

# PREDICTION_MODE='single'
# SINGLE_SMILES = 'CCO'  # 要预测的SMILES

# AD threshold
TASK_SPECIFIC_AD_PARAMS = {
    'FishAT': {
        'OPTIMAL_DENSLB': 0.2,
        'OPTIMAL_LDUB': 0.8
    },
    'FishCT': {
        'OPTIMAL_DENSLB': 0.1,
        'OPTIMAL_LDUB': 0.55
    },
    'DMCT': {
        'OPTIMAL_DENSLB': 0.01,
        'OPTIMAL_LDUB': 0.6
    },
    'DMAT': {
        'OPTIMAL_DENSLB': 0.01,
        'OPTIMAL_LDUB': 0.6
    },
    'AlgAT': {
        'OPTIMAL_DENSLB': 0.01,
        'OPTIMAL_LDUB': 0.3
    },
    'AlgCT': {
        'OPTIMAL_DENSLB': 0.01,
        'OPTIMAL_LDUB': 0.7
    }
}

# 
MODEL_PATH = '/.pth'  
TRAINING_DATA_PATH = 'data/AquaTox.csv'  # path of training data

# hyperparameter
BATCH_SIZE = 256  
AD_BATCH_SIZE = 10000  # Batch size for domain calculation (to avoid memory overflow)
# Hyperparameter of MTL-GCN（Must be identical to those used during training in AqTox_MTL_GCN.py）
MODEL_ARGS = {
    'in_feats': 40,
    'rgcn_hidden_feats': [256, 128],
    'n_tasks': 6,
    'classifier_hidden_feats': 128,
    'rgcn_drop_out': 0., 
    'dropout': 0.3,       
    'loop': True
}

# List of task names (Must be consistent with training - The order must be identical to select_task_list in AqTox_MTL_GCN.py)
TASK_NAMES = ['FishAT', 'DMAT', 'AlgAT', 'FishCT', 'DMCT', 'AlgCT']  

GRAPH_ARGS = {
    'atom_data_field': 'atom',
    'bond_data_field': 'etype'
}

def construct_molecule_graph(smiles):
    from utils.build_dataset import construct_RGCN_bigraph_from_smiles

    try:
        g = construct_RGCN_bigraph_from_smiles(smiles)
        return g
    except Exception as e:
        raise ValueError(f"failed with constructed MG - SMILES: {smiles}, error: {str(e)}")

class MTLScrPredictor:

    def __init__(self, model_path, model_args=None):
        self.model_path = model_path
        self.device = device
        self.model_args = model_args or MODEL_ARGS
        self.task_names = TASK_NAMES

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        print(f" loading MTL-GCN: {self.model_path}")
        print(f" device: {self.device}")

        self.model = MGA(
            in_feats=self.model_args['in_feats'],
            rgcn_hidden_feats=self.model_args['rgcn_hidden_feats'],
            n_tasks=self.model_args['n_tasks'],
            classifier_hidden_feats=self.model_args['classifier_hidden_feats'],
            rgcn_drop_out=self.model_args['rgcn_drop_out'],
            dropout=self.model_args['dropout'],  
            loop=self.model_args['loop']
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f" loading checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            self.model.load_state_dict(checkpoint)
            print(f" loading status dict")

        self.model.to(self.device)
        self.model.eval()

        print(f" Successfully loading MTL-GCN (device: {self.device})")

    def _validate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES"

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return True, canonical_smiles
        except Exception as e:
            return False, f"SMILES error: {str(e)}"

    def predict_single(self, smiles):
        is_valid, result = self._validate_smiles(smiles)
        if not is_valid:
            return {'smiles': smiles, 'error': result}

        canonical_smiles = result

        try:
            g = construct_molecule_graph(canonical_smiles)
            bg = dgl.batch([g]).to(self.device)
            atom_feats = bg.ndata[GRAPH_ARGS['atom_data_field']].float().to(self.device)  # 确保float类型
            bond_feats = bg.edata[GRAPH_ARGS['bond_data_field']].long().to(self.device)   # 确保long类型

            with torch.no_grad():
                predictions = self.model(bg, atom_feats, bond_feats)
                predictions = torch.sigmoid(predictions).cpu().numpy()[0]  # 应用sigmoid并转换为numpy

            result_dict = {
                'smiles': smiles,
                'canonical_smiles': canonical_smiles,
                'error': None
            }

            for i, task_name in enumerate(self.task_names):
                probability = float(predictions[i])
                prediction = int(probability > 0.5)
                result_dict[f'{task_name}_prediction'] = prediction
                result_dict[f'{task_name}_probability'] = probability
                result_dict[f'{task_name}_label'] = '有毒' if prediction == 1 else '无毒'

            return result_dict

        except Exception as e:
            return {'smiles': smiles, 'error': f"预测错误: {str(e)}"}

def predict_on_input_file(input_file_path, model_path=None, output_file_path=None):
    """
    Args:
        input_file_path: （Excel）
        model_path
        output_file_path
    """
    print("Perfoming multi-task prediction...")

    if not os.path.exists(input_file_path):
        print(f"file is not existing: {input_file_path}")
        return

    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        print(f"model is not existing: {model_path}")
        return

    if output_file_path is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}_predicted.xlsx"

    print(f"input : {input_file_path}")
    print(f"model: {model_path}")
    print(f"output: {output_file_path}")
    print(f"devie: {device}")

    try:
    
        model = MGA(
            in_feats=MODEL_ARGS['in_feats'],
            rgcn_hidden_feats=MODEL_ARGS['rgcn_hidden_feats'],
            n_tasks=MODEL_ARGS['n_tasks'],
            classifier_hidden_feats=MODEL_ARGS['classifier_hidden_feats'],
            rgcn_drop_out=MODEL_ARGS['rgcn_drop_out'],
            dropout=MODEL_ARGS['dropout'], 
            loop=MODEL_ARGS['loop']
        )

        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"loading checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"loading model dict")

        model.to(device)
        model.eval()

        print(f"Successfully loading MTL-GCN model（device: {device}）")
    except Exception as e:
        print(f"failed with MTL-GCN: {str(e)}")
        return

    try:
        df = pd.read_excel(input_file_path, dtype={'Canonical smiles': str})
        print(f"Sample number: {len(df)}")

        required_columns = ['Canonical smiles']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Input data is missing required columns: {missing_columns}")
            return

    except Exception as e:
        print(f"failed with loading data: {str(e)}")
        return


    result_df = df[['Canonical smiles']].copy() 
    result_df['error'] = ''

    for task_name in TASK_NAMES:
        result_df[f'{task_name}_prediction'] = -1      
        result_df[f'{task_name}_probability'] = 0.0    
        result_df[f'{task_name}_in_AD'] = False        

    print("Multi-task predicting...")
    valid_indices = []
    valid_canonical_smiles = []

    for idx, row in df.iterrows():
        smiles = row['Canonical smiles']

        if pd.isna(smiles):
            result_df.loc[idx, 'error'] = "SMILES is Nan"
            continue
        smiles = str(smiles).strip()

        if smiles == '' or smiles == 'nan':
            result_df.loc[idx, 'error'] = "SMILES is Nan"
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result_df.loc[idx, 'error'] = "Invalid SMILES"
                continue

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            valid_indices.append(idx)
            valid_canonical_smiles.append(canonical_smiles)

        except Exception as e:
            result_df.loc[idx, 'error'] = f"SMILES error: {str(e)}"

    print(f"Valid sample: {len(valid_indices)}/{len(df)}")

    if len(valid_indices) == 0:
        print("没有有效的样本可以预测")
        return

    # 8. 批量预测
    try:
        batch_size = BATCH_SIZE
        batch_list = [valid_indices[i:i+batch_size] for i in range(0, len(valid_indices), batch_size)]

        for batch_idx, batch_indices in enumerate(batch_list):
            print(f"处理批次 {batch_idx + 1}/{len(batch_list)}")

            batch_smiles = [valid_canonical_smiles[valid_indices.index(idx)] for idx in batch_indices]

            # 构建分子图
            graphs = []
            for smi in batch_smiles:
                try:
                    graphs.append(construct_molecule_graph(smi))
                except Exception as e:
                    print(f"构建图失败: {smi}, {str(e)}")
                    graphs.append(None)

            # 过滤有效图
            valid_graphs = [g for g in graphs if g is not None]
            if not valid_graphs:
                print(f"批次 {batch_idx + 1} 没有有效的分子图")
                continue

            # 批处理图
            bg = dgl.batch(valid_graphs).to(device)
            atom_feats = bg.ndata[GRAPH_ARGS['atom_data_field']].float().to(device)  # 确保float类型
            bond_feats = bg.edata[GRAPH_ARGS['bond_data_field']].long().to(device)   # 确保long类型

            # 预测
            with torch.no_grad():
                predictions = model(bg, atom_feats, bond_feats)
                predictions = torch.sigmoid(predictions).cpu().numpy()  # 应用sigmoid

                # 保存预测结果 - 简化版本
                valid_idx = 0
                for i, orig_idx in enumerate(batch_indices):
                    if graphs[i] is not None:  # 只处理有效图的结果
                        for j, task_name in enumerate(TASK_NAMES):
                            probability = float(predictions[valid_idx, j])
                            prediction = int(probability > 0.5)

                            result_df.loc[orig_idx, f'{task_name}_prediction'] = prediction
                            result_df.loc[orig_idx, f'{task_name}_probability'] = probability
                        valid_idx += 1
                    else:
                        # 为无效图设置错误信息
                        result_df.loc[orig_idx, 'error'] = "分子图构建失败"

                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("多任务预测完成")

    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        # 为所有有效索引标记错误
        for idx in valid_indices:
            result_df.loc[idx, 'error'] = f"预测错误: {str(e)}"

    # 9. 保存结果
    try:
        result_df.to_excel(output_file_path, index=False)
        print(f"结果已保存到: {output_file_path}")

        # 显示简化统计结果
        valid_predictions = result_df[result_df['error'] == '']
        errors = result_df[result_df['error'] != '']

        print(f"\n预测统计:")
        print(f"成功预测: {len(valid_predictions)}/{len(result_df)}")
        if len(errors) > 0:
            print(f"失败样本: {len(errors)}")

        if len(valid_predictions) > 0:
            # 简化的任务统计
            for task_name in TASK_NAMES:
                pred_col = f'{task_name}_prediction'
                if pred_col in valid_predictions.columns:
                    toxic_count = (valid_predictions[pred_col] == 1).sum()
                    print(f"{task_name}: {toxic_count}/{len(valid_predictions)} 有毒")

    except Exception as e:
        print(f"保存结果失败: {str(e)}")

# exp权重函数
def expWt(x, a=15, eps=1e-6):
    """指数权重函数"""
    return np.exp(-a*(1-x)/(x + eps))

EXP_WEIGHT_PARAMS = {'a': 15}

# =============================================================================
# 核心功能函数
# =============================================================================

def load_training_data_for_task(file_path, task_name):
    """为特定任务加载训练集数据"""
    print(f"为任务 {task_name} 加载训练集数据: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"训练数据文件不存在: {file_path}")

    df = pd.read_csv(file_path)

    # 检查必要列
    required_cols = ['smiles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"训练集缺少必要列: {missing_cols}")

    # 为特定任务创建标签
    if task_name in df.columns:
        df['y'] = df[task_name].fillna(0).astype(int)
        # 只保留该任务有有效数据的样本
        valid_samples = df[df[task_name].notna()]
        print(f"任务 {task_name} 有效训练样本数: {len(valid_samples)}")

        # 检查是否有足够的训练样本
        if len(valid_samples) == 0:
            raise ValueError(f"任务 {task_name} 没有有效的训练样本（所有值都是NaN）")

        # 检查SMILES列是否有空值
        null_smiles = valid_samples['smiles'].isna().sum()
        if null_smiles > 0:
            print(f"  警告: 发现 {null_smiles} 个空SMILES，将被过滤")
            valid_samples = valid_samples[valid_samples['smiles'].notna()]
            print(f"  过滤后有效样本数: {len(valid_samples)}")
    else:
        raise ValueError(f"训练集中没有找到任务列: {task_name}")

    # 准备数据
    df_clean = valid_samples[['smiles', 'y']].copy()
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean

def run_prediction(input_file, model_path, temp_output):
    """运行毒性预测"""
    print("步骤1: 运行毒性预测")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 运行预测
    predict_on_input_file(input_file, model_path, temp_output)

    if not os.path.exists(temp_output):
        raise RuntimeError("预测失败，未生成预测文件")

    print("预测完成")
    return temp_output

def calculate_ad_metrics_for_task(df_train, df_query, task_name, batch_size=10000):
    """为特定任务计算应用域指标（分批处理以避免内存溢出）

    Args:
        df_train: 训练数据
        df_query: 查询数据
        task_name: 任务名称
        batch_size: 每批处理的样本数，默认10000
    """
    print(f"步骤2: 为任务 {task_name} 计算应用域指标")

    try:
        # 检查训练数据
        if len(df_train) == 0:
            raise ValueError(f"任务 {task_name} 没有有效的训练数据")

        print(f"  训练集样本数: {len(df_train)}")
        print(f"  查询集样本数: {len(df_query)}")

        # 创建NSG对象
        nsg = NSG(df_train, yCol='y', smiCol='smiles')

        # 计算分子指纹相似性
        print(f"  计算分子指纹相似性...")
        nsg.calcPairwiseSimilarityWithFp('MACCS_keys')

        # 只提取SMILES列用于应用域计算，并确保数据类型正确
        df_query_smiles = df_query[['Canonical smiles']].copy()

        # 保存原始索引
        original_index = df_query_smiles.index

        # 确保SMILES列是字符串类型
        df_query_smiles['Canonical smiles'] = df_query_smiles['Canonical smiles'].astype(str)

        # 重置索引以避免索引不匹配问题
        df_query_smiles_reset = df_query_smiles.reset_index(drop=True)

        # 分批处理以避免内存溢出
        total_samples = len(df_query_smiles_reset)
        num_batches = (total_samples + batch_size - 1) // batch_size

        print(f"  分批处理: 总样本数 {total_samples}, 批次大小 {batch_size}, 总批次数 {num_batches}")

        all_ad_metrics = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)

            print(f"  处理批次 {batch_idx + 1}/{num_batches} (样本 {start_idx}-{end_idx})...")

            # 获取当前批次的数据
            df_batch = df_query_smiles_reset.iloc[start_idx:end_idx]

            # 生成查询-训练相似性矩阵
            dfQTSM_batch = nsg.genQTSM(df_batch, 'Canonical smiles')

            # 计算应用域指标（使用exp权重函数）
            ad_metrics_batch = nsg.queryADMetrics(
                dfQTSM_batch,
                wtFunc1=expWt,
                kw1=EXP_WEIGHT_PARAMS,
                wtFunc2=expWt,
                kw2=EXP_WEIGHT_PARAMS,
                code='|exp'
            )

            all_ad_metrics.append(ad_metrics_batch)

            # 清理内存
            del dfQTSM_batch
            del ad_metrics_batch

        # 合并所有批次的结果
        print(f"  合并所有批次的结果...")
        ad_metrics = pd.concat(all_ad_metrics, axis=0, ignore_index=True)

        # 为指标添加任务前缀
        ad_metrics_renamed = {}
        for col in ad_metrics.columns:
            ad_metrics_renamed[f'{task_name}_{col}'] = ad_metrics[col]

        ad_metrics_df = pd.DataFrame(ad_metrics_renamed)

        # 恢复原始索引
        ad_metrics_df.index = original_index

        print(f"任务 {task_name} 应用域指标计算完成")
        return ad_metrics_df

    except Exception as e:
        print(f"  ❌ 计算应用域指标时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def apply_multi_task_ad_criteria(df_query):
    """为多任务应用应用域判断标准（使用任务特定阈值）"""
    print("步骤3: 多任务应用域判断（使用任务特定阈值）")

    if not HAS_ADSAL:
        print("⚠️ 应用域判断功能不可用，跳过应用域分析")
        # 为每个任务添加默认的应用域信息
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_in_applicability_domain'] = True
            df_query[f'{task_name}_ad_reason'] = "应用域判断功能不可用"
        return df_query

    df_result = df_query.copy()

    # 为每个任务计算应用域
    for task_name in TASK_NAMES:
        # 获取该任务特定的阈值
        if task_name not in TASK_SPECIFIC_AD_PARAMS:
            raise ValueError(f"任务 {task_name} 未定义应用域参数")

        task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
        task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']

        print(f"\n处理任务: {task_name}")
        print(f"  相似性密度阈值 (densLB): {task_densLB}")
        print(f"  局域不连续性阈值 (LdUB): {task_LdUB}")

        try:
            # 加载该任务的训练数据
            df_train_task = load_training_data_for_task(TRAINING_DATA_PATH, task_name)

            # 计算该任务的应用域指标（使用分批处理）
            ad_metrics_task = calculate_ad_metrics_for_task(df_train_task, df_query, task_name, batch_size=AD_BATCH_SIZE)

            # 合并应用域指标
            for col in ad_metrics_task.columns:
                df_result[col] = ad_metrics_task[col]

            # 应用域判断条件
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

                # 添加应用域判断原因
                def get_task_ad_reason(row, task, dens_threshold, ld_threshold):
                    density_val = row[f'{task}_ad_density_value']
                    ld_val = row[f'{task}_ad_ld_value']

                    if row[f'{task}_in_applicability_domain']:
                        return f"{task}: 在应用域内"
                    else:
                        reasons = []
                        if density_val < dens_threshold:
                            reasons.append(f"相似性密度({density_val:.3f}) < 阈值({dens_threshold})")
                        if ld_val > ld_threshold:
                            reasons.append(f"局域不连续性({ld_val:.3f}) > 阈值({ld_threshold})")
                        return f"{task}: " + "; ".join(reasons)

                df_result[f'{task_name}_ad_reason'] = df_result.apply(
                    lambda row: get_task_ad_reason(row, task_name, task_densLB, task_LdUB), axis=1
                )

                # 统计该任务的应用域结果
                total_compounds = len(df_result)
                in_domain_count = df_result[f'{task_name}_in_applicability_domain'].sum()
                print(f"  {task_name} 应用域内: {in_domain_count}/{total_compounds} ({in_domain_count/total_compounds*100:.1f}%)")

        except Exception as e:
            print(f"  ⚠️ 任务 {task_name} 应用域计算失败: {str(e)}")
            # 设置默认值
            df_result[f'{task_name}_in_applicability_domain'] = False
            df_result[f'{task_name}_ad_reason'] = f"应用域计算失败: {str(e)}"

    return df_result

def apply_simplified_multi_task_ad_criteria(df_query):
    """简化的多任务应用域判断 - 只保留必要信息（使用任务特定阈值）"""
    print("步骤3: 简化多任务应用域判断（使用任务特定阈值）")

    if not HAS_ADSAL:
        print("⚠️ 应用域判断功能不可用，跳过应用域分析")
        # 为每个任务添加默认的应用域信息
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_in_AD'] = True
        return df_query

    df_result = df_query.copy()

    # 检查error列是否存在
    if 'error' not in df_result.columns:
        print("⚠️ 数据中没有error列，假设所有样本都有效")
        df_valid = df_result.copy()
        valid_mask = pd.Series([True] * len(df_result), index=df_result.index)
    else:
        # 将NaN视为空字符串（有效样本）
        df_result['error'] = df_result['error'].fillna('')

        # 只对有效的样本（没有错误的）进行应用域计算
        valid_mask = df_result['error'] == ''
        df_valid = df_result[valid_mask].copy()

        print(f"总样本数: {len(df_result)}, 有效样本数: {len(df_valid)}")

    if len(df_valid) == 0:
        print("⚠️ 没有有效样本可以进行应用域计算")
        return df_result

    # 为每个任务计算应用域
    for task_name in TASK_NAMES:
        # 获取该任务特定的阈值
        if task_name not in TASK_SPECIFIC_AD_PARAMS:
            raise ValueError(f"任务 {task_name} 未定义应用域参数")

        task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
        task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']

        print(f"\n处理任务: {task_name}")
        print(f"  相似性密度阈值 (densLB): {task_densLB}")
        print(f"  局域不连续性阈值 (LdUB): {task_LdUB}")

        try:
            # 加载该任务的训练数据
            df_train_task = load_training_data_for_task(TRAINING_DATA_PATH, task_name)

            # 只对有效样本计算该任务的应用域指标（使用分批处理）
            ad_metrics_task = calculate_ad_metrics_for_task(df_train_task, df_valid, task_name, batch_size=AD_BATCH_SIZE)

            # 应用域判断条件
            density_col = f'{task_name}_simiDensity|exp'
            ld_col = f'{task_name}_simiWtLD_w|exp'

            if density_col in ad_metrics_task.columns and ld_col in ad_metrics_task.columns:
                ad_condition = (
                    (ad_metrics_task[density_col] >= task_densLB) &
                    (ad_metrics_task[ld_col] <= task_LdUB)
                )

                # 将应用域结果合并回原始DataFrame（使用索引对齐）
                df_result.loc[valid_mask, f'{task_name}_in_AD'] = ad_condition.values

                # 统计该任务的应用域结果
                total_compounds = len(df_valid)
                in_domain_count = ad_condition.sum()
                print(f"  {task_name} 应用域内: {in_domain_count}/{total_compounds} ({in_domain_count/total_compounds*100:.1f}%)")

        except Exception as e:
            print(f"  ⚠️ 任务 {task_name} 应用域计算失败: {str(e)}")
            # 打印详细的错误堆栈信息
            import traceback
            print(f"  详细错误信息:")
            traceback.print_exc()
            # 对有效样本设置默认值False，无效样本保持原值
            df_result.loc[valid_mask, f'{task_name}_in_AD'] = False

    return df_result

def generate_summary(df_result):
    """生成简化的多任务结果摘要"""
    print("\n预测结果摘要:")
    print("=" * 60)

    # 只显示前5个样本的详细信息
    display_count = min(5, len(df_result))

    for idx in range(display_count):
        row = df_result.iloc[idx]
        print(f"\n样本 {idx+1}:")
        print(f"  SMILES: {row['Canonical smiles']}")

        if row.get('error', ''):
            print(f"  错误: {row['error']}")
            continue

        print(f"  预测结果:")
        for task_name in TASK_NAMES:
            pred = row.get(f'{task_name}_prediction', -1)
            prob = row.get(f'{task_name}_probability', 0)
            in_ad = row.get(f'{task_name}_in_AD', False)

            if pred != -1:
                status = "有毒" if pred == 1 else "无毒"
                ad_status = "域内" if in_ad else "域外"
                print(f"    {task_name}: {status} ({prob:.3f}) - {ad_status}")

    if len(df_result) > display_count:
        print(f"\n... 还有 {len(df_result) - display_count} 个样本")

    # 总体统计
    valid_samples = df_result[df_result['error'] == '']
    if len(valid_samples) > 0:
        print(f"\n总体统计:")
        for task_name in TASK_NAMES:
            pred_col = f'{task_name}_prediction'
            ad_col = f'{task_name}_in_AD'

            if pred_col in valid_samples.columns:
                toxic_count = (valid_samples[pred_col] == 1).sum()
                if ad_col in valid_samples.columns:
                    in_ad_count = valid_samples[ad_col].sum()
                    print(f"  {task_name}: {toxic_count}/{len(valid_samples)} 有毒, {in_ad_count}/{len(valid_samples)} 域内")
                else:
                    print(f"  {task_name}: {toxic_count}/{len(valid_samples)} 有毒")

    print("=" * 60)

def run_single_prediction_with_ad():
    """运行单个SMILES预测并进行应用域判断"""
    print("=" * 60)
    print("单个SMILES多任务预测与应用域判断")
    print("=" * 60)

    print(f"SMILES: {SINGLE_SMILES}")
    print("使用任务特定的应用域阈值")
    print()

    try:
        # 步骤1: 多任务毒性预测
        predictor = MTLScrPredictor(MODEL_PATH)
        prediction_result = predictor.predict_single(SINGLE_SMILES)

        if prediction_result.get('error'):
            print(f"预测失败: {prediction_result['error']}")
            return

        print("步骤1: 多任务毒性预测完成")
        for task_name in TASK_NAMES:
            pred_label = prediction_result.get(f'{task_name}_label', 'N/A')
            pred_prob = prediction_result.get(f'{task_name}_probability', 0)
            print(f"  {task_name}: {pred_label} (概率: {pred_prob:.4f})")

        # 步骤2: 创建查询数据
        query_data = {
            'Canonical smiles': [prediction_result['canonical_smiles']],
            'compound_name': [f"查询化合物_{SINGLE_SMILES[:10]}"]
        }
        df_query = pd.DataFrame(query_data)

        # 添加多任务预测结果到查询数据
        for task_name in TASK_NAMES:
            df_query[f'{task_name}_prediction'] = prediction_result.get(f'{task_name}_prediction', -1)
            df_query[f'{task_name}_probability'] = prediction_result.get(f'{task_name}_probability', 0.0)
            df_query[f'{task_name}_label'] = prediction_result.get(f'{task_name}_label', 'N/A')

        # 步骤3: 多任务应用域判断（使用任务特定阈值）
        df_final = apply_multi_task_ad_criteria(df_query)

        # 显示结果
        print("\n" + "=" * 60)
        print("完整多任务分析结果:")
        print("=" * 60)

        row = df_final.iloc[0]

        print(f"SMILES: {SINGLE_SMILES}")
        print(f"标准SMILES: {row['Canonical smiles']}")

        print(f"\n多任务毒性预测:")
        for task_name in TASK_NAMES:
            pred_label = row.get(f'{task_name}_label', 'N/A')
            pred_prob = row.get(f'{task_name}_probability', 0)
            in_ad = row.get(f'{task_name}_in_applicability_domain', False)
            ad_status = "应用域内" if in_ad else "应用域外"

            # 获取该任务的阈值
            if task_name not in TASK_SPECIFIC_AD_PARAMS:
                raise ValueError(f"任务 {task_name} 未定义应用域参数")

            task_densLB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_DENSLB']
            task_LdUB = TASK_SPECIFIC_AD_PARAMS[task_name]['OPTIMAL_LDUB']

            density_val = row.get(f'{task_name}_ad_density_value', 0)
            ld_val = row.get(f'{task_name}_ad_ld_value', 0)

            print(f"  {task_name}: {pred_label} (概率: {pred_prob:.4f}) - {ad_status}")
            print(f"    相似性密度: {density_val:.4f} (阈值: {task_densLB})")
            print(f"    局域不连续性: {ld_val:.4f} (阈值: {task_LdUB})")

        # 给出建议
        print(f"\n建议:")
        for task_name in TASK_NAMES:
            in_ad = row.get(f'{task_name}_in_applicability_domain', False)
            if in_ad:
                suggestion = f"  {task_name}: 预测结果可信，建议采用"
            else:
                suggestion = f"  {task_name}: 预测结果可信度较低，建议实验验证"
            print(suggestion)

        # 保存结果
        output_file = f"single_prediction_result_{SINGLE_SMILES[:10].replace('/', '_')}.xlsx"
        df_final.to_excel(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

    except Exception as e:
        print(f"分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_prediction_pipeline():
    """测试预测管道的基本功能"""
    print("=" * 60)
    print("测试预测管道")
    print("=" * 60)

    # 测试SMILES列表
    test_smiles = [
        'CCO',  # 乙醇 - 简单分子
        'CC(C)O',  # 异丙醇
        'c1ccccc1',  # 苯
        'CCN(CC)CC',  # 三乙胺
        'invalid_smiles'  # 无效SMILES用于测试错误处理
    ]

    try:
        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 模型文件不存在: {MODEL_PATH}")
            print("请确保模型文件路径正确")
            return False

        print(f"✅ 模型文件存在: {MODEL_PATH}")

        # 初始化预测器
        print("初始化预测器...")
        predictor = MTLScrPredictor(MODEL_PATH)
        print("✅ 预测器初始化成功")

        # 测试单个SMILES预测
        print("\n测试单个SMILES预测:")
        for i, smiles in enumerate(test_smiles):
            print(f"\n测试 {i+1}: {smiles}")
            result = predictor.predict_single(smiles)

            if result.get('error'):
                print(f"  ❌ 预测失败: {result['error']}")
            else:
                print(f"  ✅ 预测成功")
                print(f"  标准SMILES: {result['canonical_smiles']}")
                for task_name in TASK_NAMES:
                    pred_label = result.get(f'{task_name}_label', 'N/A')
                    pred_prob = result.get(f'{task_name}_probability', 0)
                    print(f"    {task_name}: {pred_label} (概率: {pred_prob:.4f})")

        print("\n✅ 预测管道测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("MTL-scr多任务化学品毒性预测与应用域判断整合分析")
    print("=" * 60)

    print(f"预测模式: {PREDICTION_MODE}")
    print(f"模型: MTL-scr (多任务学习)")
    print(f"任务: {', '.join(TASK_NAMES)}")
    if PREDICTION_MODE == 'file':
        print(f"输入文件: {INPUT_FILE}")
        print(f"输出文件: {OUTPUT_FILE}")
    elif PREDICTION_MODE == 'single':
        print(f"单个SMILES: {SINGLE_SMILES}")
    elif PREDICTION_MODE == 'test':
        print(f"测试模式: 验证预测管道")
    print(f"使用任务特定的应用域阈值")
    print()

    try:
        # 检查应用域判断功能是否可用
        if not HAS_ADSAL:
            print("⚠️ 警告: 应用域判断功能不可用，将只进行毒性预测")

        if PREDICTION_MODE == 'test':
            # 测试模式
            print("运行预测管道测试...")
            success = test_prediction_pipeline()
            if success:
                print("\n✅ 预测管道测试通过！可以进行正式预测。")
            else:
                print("\n❌ 预测管道测试失败！请检查配置。")

        elif PREDICTION_MODE == 'single':
            if HAS_ADSAL:
                run_single_prediction_with_ad()
            else:
                # 只进行毒性预测
                predictor = MTLScrPredictor(MODEL_PATH)
                prediction_result = predictor.predict_single(SINGLE_SMILES)

                if prediction_result.get('error'):
                    print(f"预测失败: {prediction_result['error']}")
                    return

                print("多任务毒性预测结果:")
                for task_name in TASK_NAMES:
                    pred_label = prediction_result.get(f'{task_name}_label', 'N/A')
                    pred_prob = prediction_result.get(f'{task_name}_probability', 0)
                    print(f"  {task_name}: {pred_label} (概率: {pred_prob:.4f})")

        elif PREDICTION_MODE == 'file':
            # 检查输入文件
            if not os.path.exists(INPUT_FILE):
                raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

            # 步骤1: 运行毒性预测
            temp_prediction_file = 'temp_prediction.xlsx'
            run_prediction(INPUT_FILE, MODEL_PATH, temp_prediction_file)

            if HAS_ADSAL:
                # 步骤2&3: 多任务应用域判断（使用任务特定阈值）
                df_prediction = pd.read_excel(temp_prediction_file)
                print(f"\n读取预测文件: {temp_prediction_file}")
                print(f"列名: {df_prediction.columns.tolist()}")
                print(f"数据形状: {df_prediction.shape}")
                if 'error' in df_prediction.columns:
                    error_count = (df_prediction['error'] != '').sum()
                    print(f"有错误的样本数: {error_count}")

                df_final = apply_simplified_multi_task_ad_criteria(df_prediction)

                # 步骤5: 保存结果
                df_final.to_excel(OUTPUT_FILE, index=False)
                print(f"\n结果已保存到: {OUTPUT_FILE}")

                # 步骤6: 生成摘要
                generate_summary(df_final)
            else:
                # 只保存预测结果
                print(f"\n预测结果已保存到: {temp_prediction_file}")
                print("⚠️ 由于缺少adsal模块，未进行应用域判断")

            # 清理临时文件
            if HAS_ADSAL and os.path.exists(temp_prediction_file):
                os.remove(temp_prediction_file)

            print("\n分析完成！")
        else:
            raise ValueError(f"不支持的预测模式: {PREDICTION_MODE}，支持的模式: 'file', 'single', 'test'")

    except Exception as e:
        print(f"分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

        # 清理临时文件
        if os.path.exists('temp_prediction.xlsx'):
            os.remove('temp_prediction.xlsx')

        sys.exit(1)

if __name__ == "__main__":
    main()

