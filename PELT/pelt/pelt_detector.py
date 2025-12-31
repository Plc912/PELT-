"""
PELT (Pruned Exact Linear Time) 变化点检测算法
用于时序分割，检测平稳段之间的突变点
"""
import numpy as np
from typing import List, Tuple, Optional


class PELTDetector:
    """
    PELT算法实现
    用于检测时间序列中的变化点，将序列分割为平稳段
    """
    
    def __init__(
        self,
        penalty: float = 2.0 * np.log(10),
        min_segment_length: int = 2,
    ):
        """
        初始化PELT检测器
        
        Args:
            penalty: 惩罚参数，控制检测的敏感度（越大越不敏感）
            min_segment_length: 最小段长度
        """
        self.penalty = penalty
        self.min_segment_length = min_segment_length
    
    def _calculate_cost(self, data: np.ndarray, start: int, end: int) -> float:
        """
        计算段[start, end)的成本（负对数似然）
        
        Args:
            data: 数据数组
            start: 起始索引
            end: 结束索引
            
        Returns:
            成本值
        """
        segment = data[start:end]
        if len(segment) < 2:
            return 0.0
        
        # 计算均值
        mean = np.mean(segment)
        
        # 计算方差的对数似然（负值）
        variance = np.var(segment)
        if variance < 1e-10:  # 避免除以零
            variance = 1e-10
        
        n = len(segment)
        # 高斯分布的对数似然（负值作为成本）
        cost = n * np.log(2 * np.pi * variance) + n
        return cost
    
    def _find_changepoints(self, data: np.ndarray) -> List[int]:
        """
        使用动态规划找到最优变化点
        
        Args:
            data: 输入数据数组
            
        Returns:
            变化点列表（索引）
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return []
        
        # F[t] = 最小成本到时间t
        F = np.zeros(n + 1)
        F[0] = 0
        
        # R: 可能的最后变化点集合
        R = [0]
        changepoints = [0] * (n + 1)
        
        for t in range(1, n + 1):
            # 计算最小成本和最佳变化点
            best_cost = np.inf
            best_cp = 0
            
            # 只考虑R集合中的变化点
            for s in R:
                if t - s < self.min_segment_length:
                    continue
                
                cost = F[s] + self._calculate_cost(data, s, t) + self.penalty
                if cost < best_cost:
                    best_cost = cost
                    best_cp = s
            
            F[t] = best_cost
            changepoints[t] = best_cp
            
            # 剪枝：移除不可能的变化点
            R_new = [t]
            for s in R:
                if t - s >= self.min_segment_length:
                    cost_with_s = F[s] + self._calculate_cost(data, s, t) + self.penalty
                    cost_without_s = F[t]
                    # 如果保留s可能更优，则保留
                    if cost_with_s < cost_without_s + self.penalty:
                        R_new.append(s)
            R = sorted(R_new)
        
        # 回溯找到所有变化点
        cp_list = []
        t = n
        while t > 0:
            cp_list.append(changepoints[t])
            t = changepoints[t]
        
        cp_list.reverse()
        # 移除起点，只保留中间的变化点
        cp_list = [cp for cp in cp_list[1:] if cp > 0]
        
        return sorted(list(set(cp_list)))
    
    def detect(
        self,
        data: np.ndarray,
        return_segments: bool = True,
    ) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
        """
        检测变化点
        
        Args:
            data: 输入数据（一维数组）
            return_segments: 是否返回分段信息
            
        Returns:
            changepoints: 变化点列表（索引）
            segments: 分段列表 [(start, end), ...]，如果return_segments=False则为None
        """
        if len(data) < 2 * self.min_segment_length:
            return [], None if not return_segments else [(0, len(data))]
        
        changepoints = self._find_changepoints(data)
        
        segments = None
        if return_segments:
            # 构建分段
            segments = []
            if not changepoints:
                segments = [(0, len(data))]
            else:
                start = 0
                for cp in changepoints:
                    segments.append((start, cp))
                    start = cp
                segments.append((start, len(data)))
        
        return changepoints, segments
    
    def detect_anomalies(
        self,
        data: np.ndarray,
        anomaly_threshold: float = 3.0,
    ) -> Tuple[List[int], List[float]]:
        """
        检测异常点（基于变化点周围的统计特性）
        
        Args:
            data: 输入数据
            anomaly_threshold: 异常阈值（标准差倍数）
            
        Returns:
            anomaly_indices: 异常点索引列表
            anomaly_scores: 异常分数列表
        """
        changepoints, segments = self.detect(data, return_segments=True)
        
        anomaly_indices = []
        anomaly_scores = []
        
        if not segments:
            return anomaly_indices, anomaly_scores
        
        # 对每个分段计算统计特性
        for i, (start, end) in enumerate(segments):
            segment = data[start:end]
            
            if len(segment) < 2:
                continue
            
            mean = np.mean(segment)
            std = np.std(segment)
            
            if std < 1e-10:
                continue
            
            # 检测段内的异常点
            z_scores = np.abs((segment - mean) / std)
            anomalies_in_segment = np.where(z_scores > anomaly_threshold)[0]
            
            for idx in anomalies_in_segment:
                global_idx = start + idx
                anomaly_indices.append(global_idx)
                anomaly_scores.append(float(z_scores[idx]))
        
        return anomaly_indices, anomaly_scores

