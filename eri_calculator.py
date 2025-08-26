#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import logging
import chardet
from io import StringIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_file_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # 读取前10000字节用于检测
    return chardet.detect(rawdata)['encoding']

def normalize_sample_names(sample_name):
    """标准化样本名称，处理CK1-1到CK-1的转换"""
    if sample_name.startswith('CK1-'):
        return 'CK-' + sample_name.split('-')[1]
    return sample_name

def load_group_file(group_file):
    """加载分组文件"""
    groups = {}
    # 首先尝试自动检测编码
    try:
        detected_encoding = detect_file_encoding(group_file)
        if detected_encoding:
            encodings = [detected_encoding, 'utf-8-sig', 'utf-8', 'gb18030', 'latin1', 'cp1252']
        else:
            encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'latin1', 'cp1252']
    except Exception as e:
        logger.debug(f"编码自动检测失败: {str(e)}")
        encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(group_file, 'r', encoding=encoding) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        sample = normalize_sample_names(parts[0].strip())
                        group = parts[1].strip()
                        groups[sample] = group
            logger.info(f"成功使用 {encoding} 编码加载分组文件")
            return groups
        except (UnicodeDecodeError, Exception) as e:
            logger.debug(f"尝试 {encoding} 编码失败: {str(e)}")
            continue
    
    # 所有编码都失败时使用最终方案
    logger.warning("使用带错误处理的 utf-8 编码加载分组文件")
    try:
        with open(group_file, 'rb') as f:
            content = f.read().decode('utf-8', errors='replace')
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                sample = normalize_sample_names(parts[0].strip())
                group = parts[1].strip()
                groups[sample] = group
        return groups
    except Exception as e:
        logger.error(f"加载分组文件失败: {str(e)}")
        raise

def load_data_file(file_path, index_col=0):
    """加载数据文件，处理样本名称标准化"""
    # 尝试的编码列表（按优先级）
    try:
        detected_encoding = detect_file_encoding(file_path)
        if detected_encoding:
            encodings = [detected_encoding, 'utf-8-sig', 'utf-16', 'gb18030', 'latin1', 'cp1252']
        else:
            encodings = ['utf-8-sig', 'utf-16', 'gb18030', 'latin1', 'cp1252']
    except Exception as e:
        logger.debug(f"编码自动检测失败: {str(e)}")
        encodings = ['utf-8-sig', 'utf-16', 'gb18030', 'latin1', 'cp1252']
    
    # 尝试各种编码
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, sep='\t', index_col=index_col, encoding=encoding)
            # 标准化列名（样本名）
            data.columns = [normalize_sample_names(col) for col in data.columns]
            logger.info(f"成功使用 {encoding} 编码加载文件: {file_path}")
            return data
        except (UnicodeDecodeError, Exception) as e:
            logger.debug(f"尝试 {encoding} 编码失败: {str(e)}")
            continue
    
    # 所有编码都失败时的最终方案：二进制模式+错误处理
    logger.warning(f"所有编码尝试失败，使用二进制模式加载文件: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='replace')
        data = pd.read_csv(StringIO(content), sep='\t', index_col=index_col)
        data.columns = [normalize_sample_names(col) for col in data.columns]
        return data
    except Exception as e:
        logger.error(f"无法加载文件 {file_path}: {str(e)}")
        raise

def identify_numeric_columns(data):
    """识别数据框中的数值列"""
    numeric_cols = []
    for col in data.columns:
        try:
            # 尝试将列转换为数值型
            pd.to_numeric(data[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            logger.debug(f"列 '{col}' 不是数值型，已跳过")
            continue
    return numeric_cols

def calculate_csi(otu_file, groups):
    """计算群落结构偏移指数(CSI) - 通用版"""
    logger.info("Calculating Community Shift Index (CSI)...")
    
    try:
        # 加载OTU数据
        otu_data = load_data_file(otu_file)
        
        # 自动识别数值列（样本列）
        sample_cols = identify_numeric_columns(otu_data)
        if not sample_cols:
            logger.error("未找到数值列，无法计算CSI")
            return {}
            
        logger.info(f"识别到 {len(sample_cols)} 个数值列（样本）: {sample_cols}")
        
        # 过滤低丰度OTU并Hellinger转换
        otu_matrix = otu_data[sample_cols].copy()
        otu_matrix = otu_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
        otu_matrix = otu_matrix.loc[(otu_matrix > 0).any(axis=1)]
        
        if otu_matrix.empty:
            logger.error("过滤后OTU表为空！")
            return {}
            
        # Hellinger转换
        col_sums = otu_matrix.sum(axis=0)
        if any(col_sums == 0):
            logger.error("存在样本总丰度为0！")
            return {}
            
        otu_matrix = np.sqrt(otu_matrix / col_sums)
        
        # 计算Bray-Curtis距离矩阵
        dist_array = pdist(otu_matrix.T, metric='braycurtis')
        dist_matrix = pd.DataFrame(
            squareform(dist_array),
            index=sample_cols,
            columns=sample_cols
        )
        
        # 分组样本
        group_samples = {}
        for group in set(groups.values()):
            group_samples[group] = [s for s in sample_cols if groups.get(s) == group]
        
        # 检查对照组样本
        control_group = 'CK'
        if control_group not in group_samples or len(group_samples[control_group]) < 1:
            logger.error(f"对照组 '{control_group}' 无样本！")
            return {}
        
        # 获取所有距离值用于标准化
        all_distances = dist_matrix.values.flatten()
        max_distance = np.max(all_distances)
        min_distance = np.min(all_distances)
        range_distance = max_distance - min_distance + 1e-10  # 避免除以零
        
        # 计算标准化后的组内和组间距离
        csi_results = {}
        d_ck = dist_matrix.loc[group_samples[control_group], group_samples[control_group]].mean().mean()
        d_ck_norm = (d_ck - min_distance) / range_distance
        
        for group, samples in group_samples.items():
            if group == control_group:
                csi_results[group] = 0.0
            else:
                if not samples:
                    logger.warning(f"组 '{group}' 无样本，跳过")
                    csi_results[group] = np.nan
                else:
                    # 计算处理组与对照组的平均距离
                    d_group_ck = dist_matrix.loc[samples, group_samples[control_group]].mean().mean()
                    d_group_ck_norm = (d_group_ck - min_distance) / range_distance
                    
                    # 计算CSI (标准化差值，限制在0-1范围)
                    csi = np.clip(d_group_ck_norm - d_ck_norm, 0, 1)
                    csi_results[group] = round(csi, 6)  # 保留6位小数
        
        logger.debug(f"距离矩阵统计:\n{dist_matrix.describe()}")
        logger.debug(f"最大距离: {max_distance}, 最小距离: {min_distance}")
        logger.debug(f"对照组内平均距离: {d_ck} (标准化: {d_ck_norm})")
        
        return csi_results
    
    except Exception as e:
        logger.error(f"计算CSI时出错: {str(e)}", exc_info=True)
        return {}

def calculate_msi(eco_file, groups):
    """计算代谢功能偏移指数(MSI) - 通用版"""
    logger.info("Calculating Metabolic Shift Index (MSI)...")
    
    try:
        # 加载代谢数据
        eco_data = load_data_file(eco_file)
        eco_data = eco_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 自动识别数值列（样本列）
        sample_cols = identify_numeric_columns(eco_data)
        if not sample_cols:
            logger.error("未找到数值列，无法计算MSI")
            return {}
            
        logger.info(f"识别到 {len(sample_cols)} 个数值列（样本）: {sample_cols}")
        logger.info(f"识别到 {len(eco_data.index)} 个代谢指标: {eco_data.index.tolist()}")
        
        # 分组计算平均向量
        group_vectors = {}
        for group in set(groups.values()):
            group_samples = [s for s in sample_cols if groups.get(s) == group]
            if group_samples:
                group_vectors[group] = eco_data[group_samples].mean(axis=1)
        
        # 检查对照组
        if 'CK' not in group_vectors:
            logger.error("缺少对照组 'CK'")
            return {}
        
        # 计算对照组范数
        ck_vector = group_vectors['CK']
        ck_norm = np.linalg.norm(ck_vector)
        if ck_norm == 0:
            logger.warning("对照组向量范数为0，设置为小值避免除零错误")
            ck_norm = 1e-10
        
        # 计算MSI
        msi_results = {}
        for group, vector in group_vectors.items():
            if group == 'CK':
                msi_results[group] = 0.0
            else:
                distance = np.linalg.norm(vector - ck_vector)
                msi = distance / ck_norm
                msi_results[group] = round(msi, 6)
        
        return msi_results
    
    except Exception as e:
        logger.error(f"计算MSI时出错: {str(e)}", exc_info=True)
        return {}

def calculate_dsi(enzyme_file, groups):
    """计算解毒能力偏移指数(DSI) - 通用版"""
    logger.info("Calculating Detoxification Shift Index (DSI)...")
    
    try:
        # 加载酶活数据
        enzyme_data = load_data_file(enzyme_file)
        enzyme_data = enzyme_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 自动识别数值列（样本列）
        sample_cols = identify_numeric_columns(enzyme_data)
        if not sample_cols:
            logger.error("未找到数值列，无法计算DSI")
            return {}
            
        logger.info(f"识别到 {len(sample_cols)} 个数值列（样本）: {sample_cols}")
        logger.info(f"识别到 {len(enzyme_data.index)} 个酶指标: {enzyme_data.index.tolist()}")
        
        # 分组计算平均值
        group_means = {}
        for group in set(groups.values()):
            group_samples = [s for s in sample_cols if groups.get(s) == group]
            if group_samples:
                group_means[group] = enzyme_data[group_samples].mean(axis=1)
        
        # 检查对照组
        if 'CK' not in group_means:
            logger.error("缺少对照组 'CK'")
            return {}
        
        # 计算DSI
        dsi_results = {}
        ck_means = group_means['CK']
        
        for group, means in group_means.items():
            if group == 'CK':
                dsi_results[group] = 0.0
            else:
                changes = []
                for enzyme in enzyme_data.index:
                    ck_val = ck_means.loc[enzyme]
                    group_val = means.loc[enzyme]
                    if isinstance(ck_val, (int, float)) and ck_val > 0:
                        change = (ck_val - group_val) / ck_val
                        changes.append(max(change, 0))
                
                dsi = np.mean(changes) if changes else 0
                dsi_results[group] = round(dsi, 6)
        
        return dsi_results
    
    except Exception as e:
        logger.error(f"计算DSI时出错: {str(e)}", exc_info=True)
        return {}

def calculate_shsi(spc_file, groups):
    """计算土壤健康偏移指数(SHSI) - 通用版"""
    logger.info("Calculating Soil Health Shift Index (SHSI)...")
    
    try:
        # 加载土壤理化数据
        spc_data = load_data_file(spc_file)
        spc_data = spc_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 自动识别数值列（样本列）
        sample_cols = identify_numeric_columns(spc_data)
        if not sample_cols:
            logger.error("未找到数值列，无法计算SHSI")
            return {}
            
        logger.info(f"识别到 {len(sample_cols)} 个数值列（样本）: {sample_cols}")
        logger.info(f"识别到 {len(spc_data.index)} 个土壤指标: {spc_data.index.tolist()}")
        
        # 分组计算平均值
        group_means = {}
        for group in set(groups.values()):
            group_samples = [s for s in sample_cols if groups.get(s) == group]
            if group_samples:
                group_means[group] = spc_data[group_samples].mean(axis=1)
        
        # 检查对照组
        if 'CK' not in group_means:
            logger.error("缺少对照组 'CK'")
            return {}
        
        # 计算SHSI
        shsi_results = {}
        ck_means = group_means['CK']
        
        for group, means in group_means.items():
            if group == 'CK':
                shsi_results[group] = 0.0
            else:
                changes = []
                for param in spc_data.index:
                    ck_val = ck_means.loc[param]
                    group_val = means.loc[param]
                    if isinstance(ck_val, (int, float)) and ck_val > 0:
                        change = abs((group_val - ck_val) / ck_val)
                        changes.append(change)
                
                shsi = np.mean(changes) if changes else 0
                shsi_results[group] = round(shsi, 6)
        
        return shsi_results
    
    except Exception as e:
        logger.error(f"计算SHSI时出错: {str(e)}", exc_info=True)
        return {}

def main():
    parser = argparse.ArgumentParser(description='Calculate Ecological Risk Index (ERI)')
    parser.add_argument('-enzyme', required=True, help='Enzyme activity file (TSV format)')
    parser.add_argument('-otu', required=True, help='OTU abundance file (TSV format)')
    parser.add_argument('-spc', required=True, help='Soil physicochemical properties file (TSV format)')
    parser.add_argument('-eco', required=True, help='Metabolic function file (TSV format)')
    parser.add_argument('-group', required=True, help='Group mapping file (sample<tab>group)')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 加载分组文件
    groups = load_group_file(args.group)
    logger.info(f"Loaded group mapping for {len(groups)} samples")
    
    # 计算各组分指数
    csi_results = calculate_csi(args.otu, groups)
    msi_results = calculate_msi(args.eco, groups)
    dsi_results = calculate_dsi(args.enzyme, groups)
    shsi_results = calculate_shsi(args.spc, groups)
    
    # 计算ERI
    eri_results = {}
    all_groups = sorted(set(groups.values()))
    
    for group in all_groups:
        csi = csi_results.get(group, np.nan)
        msi = msi_results.get(group, np.nan)
        dsi = dsi_results.get(group, np.nan)
        shsi = shsi_results.get(group, np.nan)
        
        if np.isnan(csi) or np.isnan(msi) or np.isnan(dsi) or np.isnan(shsi):
            eri = np.nan
            calc_formula = "无法计算"
        else:
            eri = 0.25 * (csi + msi + dsi + shsi)
            calc_formula = f"0.25*({csi:.3f}+{msi:.3f}+{dsi:.3f}+{shsi:.3f})"
        
        eri_results[group] = {
            'CSI': csi,
            'MSI': msi,
            'DSI': dsi,
            'SHSI': shsi,
            'ERI_calc': calc_formula,
            'ERI': eri
        }
    
    # 创建结果DataFrame
    result_data = []
    for group in all_groups:
        result_data.append({
            'Group': group,
            'CSI': eri_results[group]['CSI'],
            'MSI': eri_results[group]['MSI'],
            'DSI': eri_results[group]['DSI'],
            'SHSI': eri_results[group]['SHSI'],
            'ERI_calc': eri_results[group]['ERI_calc'],
            'ERI': eri_results[group]['ERI']
        })
    
    result_df = pd.DataFrame(result_data).set_index('Group')
    result_df.to_csv(args.output, sep='\t', float_format='%.4f', na_rep='NaN')
    
    # 打印结果
    print("\nEcological Risk Index (ERI) Results:")
    print(result_df[['CSI', 'MSI', 'DSI', 'SHSI', 'ERI']].to_string())
    
    if result_df.isna().any().any():
        print("\n警告: 某些指标计算结果为NaN，请检查输入数据和日志")

if __name__ == '__main__':
    main()