import pandas as pd
import numpy as np
import os
from skfp.fingerprints import PubChemFingerprint

def calculate_pubchem_fingerprint(smiles_list):
    """
    批量计算PubChem分子指纹
    
    Parameters:
    smiles_list (list): SMILES字符串列表
    
    Returns:
    numpy.ndarray: 指纹矩阵 (n_molecules x 881)，如果计算失败则返回None
    """
    try:
        # 创建PubChem指纹计算器
        fp = PubChemFingerprint()
        
        # 批量计算指纹
        fingerprints = fp.transform(smiles_list)
        
        return fingerprints
    
    except Exception as e:
        print(f"计算PubChem指纹时出错: {str(e)}")
        return None

def calculate_single_pubchem_fingerprint(smiles):
    """
    计算单个分子的PubChem指纹
    
    Parameters:
    smiles (str): SMILES字符串
    
    Returns:
    list: 二进制指纹列表 (881位)，如果SMILES无效则返回None
    """
    try:
        # 创建PubChem指纹计算器
        fp = PubChemFingerprint()
        
        # 计算单个分子的指纹
        fingerprint = fp.transform([smiles])
        
        # 返回第一个（也是唯一一个）分子的指纹
        return fingerprint[0].tolist()
    
    except Exception as e:
        print(f"计算指纹时出错 (SMILES: {smiles}): {str(e)}")
        return None

def process_excel_file(input_file, output_file=None, smiles_column='SMILES', batch_processing=True):
    """
    处理Excel文件，计算PubChem指纹并保存结果
    
    Parameters:
    input_file (str): 输入Excel文件路径
    output_file (str): 输出Excel文件路径，如果为None则自动生成
    smiles_column (str): SMILES列的名称
    batch_processing (bool): 是否使用批处理模式（推荐）
    """
    
    # PubChem指纹固定为881位
    n_bits = 881
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取Excel文件
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        raise Exception(f"读取Excel文件失败: {str(e)}")
    
    # 检查SMILES列是否存在
    if smiles_column not in df.columns:
        print(f"可用的列名: {list(df.columns)}")
        raise ValueError(f"未找到SMILES列: {smiles_column}")
    
    print(f"找到 {len(df)} 行数据")
    print(f"SMILES列名: {smiles_column}")
    print(f"PubChem指纹位数: {n_bits}")
    
    if batch_processing:
        print("使用批处理模式计算指纹...")
        # 批处理模式 - 更高效
        
        # 准备SMILES列表，处理空值
        smiles_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            if pd.isna(smiles) or smiles == '':
                smiles_list.append("C")  # 使用简单分子作为占位符
            else:
                smiles_list.append(smiles)
            valid_indices.append(idx)
        
        # 批量计算指纹
        fingerprints_matrix = calculate_pubchem_fingerprint(smiles_list)
        
        if fingerprints_matrix is None:
            raise Exception("批量计算PubChem指纹失败")
        
        # 处理空的SMILES，将其指纹设为全0
        fingerprints = []
        successful_count = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            smiles = row[smiles_column]
            if pd.isna(smiles) or smiles == '':
                print(f"第 {idx+1} 行: SMILES为空")
                fingerprints.append([0] * n_bits)
            else:
                fingerprints.append(fingerprints_matrix[i].tolist())
                successful_count += 1
    
    else:
        print("使用逐个处理模式计算指纹...")
        # 逐个处理模式
        fingerprints = []
        successful_count = 0
        
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            
            if pd.isna(smiles) or smiles == '':
                print(f"第 {idx+1} 行: SMILES为空")
                fingerprints.append([0] * n_bits)
            else:
                fp = calculate_single_pubchem_fingerprint(smiles)
                if fp is not None:
                    fingerprints.append(fp)
                    successful_count += 1
                else:
                    fingerprints.append([0] * n_bits)
            
            # 显示进度
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 行")
    
    print(f"成功计算了 {successful_count}/{len(df)} 个分子的指纹")
    
    # 创建新的DataFrame，包含原始数据和指纹
    result_df = df.copy()
    
    # 将指纹数据添加到DataFrame中，每一位占一列
    fp_columns = [f'PubChem_{i:03d}' for i in range(n_bits)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_columns, index=df.index)
    
    # 合并原始数据和指纹数据
    result_df = pd.concat([result_df, fp_df], axis=1)
    
    # 生成输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_PubChem.xlsx"
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    try:
        result_df.to_excel(output_file, index=False)
        print(f"成功保存! 输出文件包含 {len(result_df.columns)} 列")
        print(f"原始数据列数: {len(df.columns)}")
        print(f"PubChem指纹列数: {n_bits}")
    except Exception as e:
        raise Exception(f"保存Excel文件失败: {str(e)}")
    
    return result_df

def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    input_file = "/.xlsx"  # 输入文件路径
    smiles_column = "smiles"  # SMILES列的名称
    batch_processing = True  # 是否使用批处理模式（推荐，速度更快）
    
    try:
        # 处理文件
        result_df = process_excel_file(
            input_file=input_file,
            smiles_column=smiles_column,
            batch_processing=batch_processing
        )
        
        print(f"\n处理完成!")
        print(f"结果包含 {len(result_df)} 行, {len(result_df.columns)} 列")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
