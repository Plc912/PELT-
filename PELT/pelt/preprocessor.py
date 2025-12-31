"""
数据预处理模块
用于加载和预处理时间序列数据
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class Preprocessor:
    """
    时间序列数据预处理器
    """
    
    def load_csv(
        self,
        filepath: str,
        value_col: str = None,
        timestamp_col: str = "timestamp",
        skip_header: int = 0,
        delimiter: str = ",",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """
        从CSV文件加载数据
        
        Args:
            filepath: CSV文件路径
            value_col: 要加载的数值列名（None则自动检测）
            timestamp_col: 时间戳列名
            skip_header: 跳过的头部行数
            delimiter: 分隔符
            
        Returns:
            data: 数据数组（一维）
            timestamps: 时间戳数组（可选）
            column_name: 使用的列名
        """
        # 读取CSV文件
        df = pd.read_csv(filepath, sep=delimiter, skiprows=skip_header)
        
        # 如果timestamp列存在，按时间排序
        if timestamp_col in df.columns:
            df = df.sort_values(by=timestamp_col)
        
        # 确定要使用的数值列
        if value_col is None:
            # 自动检测第一个数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in numeric_cols:
                numeric_cols.remove(timestamp_col)
            # 排除标签列
            if 'label' in numeric_cols:
                numeric_cols.remove('label')
            if 'anomaly' in numeric_cols:
                numeric_cols.remove('anomaly')
            if numeric_cols:
                value_col = numeric_cols[0]
            else:
                raise ValueError("No numeric column found in CSV file")
        
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in CSV file")
        
        # 提取数据
        data = df[value_col].values
        timestamps = df[timestamp_col].values if timestamp_col in df.columns else None
        
        return data, timestamps, value_col

