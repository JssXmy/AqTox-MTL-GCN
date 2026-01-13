import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import os

def calculate_ecfp4_fingerprint(smiles, radius=2, n_bits=2048):
    """
    计算ECFP4分子指纹
    
    Parameters:
    smiles (str): SMILES字符串
    radius (int): 指纹半径，ECFP4使用半径2
    n_bits (int): 指纹位数
    
    Returns:
    list: 二进制指纹列表，如果SMILES无效则返回None
    """
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告: 无法解析SMILES: {smiles}")
            return None
        
        # 计算ECFP4指纹 (半径=2, 对应ECFP4)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        
        # 转换为二进制列表
        fp_bits = [int(x) for x in fp.ToBitString()]
        return fp_bits
    
    except Exception as e:
        print(f"计算指纹时出错 (SMILES: {smiles}): {str(e)}")
        return None

def process_excel_file(input_file, output_file=None, smiles_column='SMILES', n_bits=2048):
    """
    处理Excel文件，计算ECFP4指纹并保存结果
    
    Parameters:
    input_file (str): 输入Excel文件路径
    output_file (str): 输出Excel文件路径，如果为None则自动生成
    smiles_column (str): SMILES列的名称
    n_bits (int): 指纹位数
    """
    
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
    
    # 为每一行计算ECFP4指纹
    fingerprints = []
    successful_count = 0
    
    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        
        if pd.isna(smiles) or smiles == '':
            print(f"第 {idx+1} 行: SMILES为空")
            fingerprints.append([0] * n_bits)  # 用零填充
        else:
            fp = calculate_ecfp4_fingerprint(smiles, n_bits=n_bits)
            if fp is not None:
                fingerprints.append(fp)
                successful_count += 1
            else:
                fingerprints.append([0] * n_bits)  # 用零填充失败的情况
        
        # 显示进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 行")
    
    print(f"成功计算了 {successful_count}/{len(df)} 个分子的指纹")
    
    # 创建新的DataFrame，包含原始数据和指纹
    result_df = df.copy()
    
    # 将指纹数据添加到DataFrame中，每一位占一列
    fp_columns = [f'ECFP4_{i:04d}' for i in range(n_bits)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_columns, index=df.index)
    
    # 合并原始数据和指纹数据
    result_df = pd.concat([result_df, fp_df], axis=1)
    
    # 生成输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_ECFP4.xlsx"
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    try:
        result_df.to_excel(output_file, index=False)
        print(f"成功保存! 输出文件包含 {len(result_df.columns)} 列")
        print(f"原始数据列数: {len(df.columns)}")
        print(f"ECFP4指纹列数: {n_bits}")
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
    n_bits = 2048  # ECFP4指纹位数 (可以调整为1024, 2048, 4096等)
    
    try:
        # 处理文件
        result_df = process_excel_file(
            input_file=input_file,
            smiles_column=smiles_column,
            n_bits=n_bits
        )
        
        print(f"\n处理完成!")
        print(f"结果包含 {len(result_df)} 行, {len(result_df.columns)} 列")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
