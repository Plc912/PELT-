"""
生成PELT算法测试数据
包含平稳段和突变点，以及明显的异常点
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_time_series_with_changepoints(
    n_samples: int = 2000,
    n_changepoints: int = 8,
    noise_level: float = 0.5
) -> tuple:
    """
    生成带变化点的时间序列数据，包含明显的异常点
    
    Args:
        n_samples: 样本数量
        n_changepoints: 变化点数
        noise_level: 噪声水平（增加以产生更真实的波动）
        
    Returns:
        data: 数据数组
        changepoints: 真实变化点位置
        anomalies: 异常点位置
    """
    # 随机生成变化点位置，确保段之间有足够的间隔
    np.random.seed(42)
    changepoints = sorted(np.random.choice(
        range(100, n_samples - 100),
        size=n_changepoints,
        replace=False
    ))
    changepoints = [0] + changepoints + [n_samples]
    
    data = np.zeros(n_samples)
    true_anomalies = []
    
    # 定义不同的段类型和特征，使变化更明显
    # 确保段之间有足够大的均值差异（至少15-25个单位）
    segment_configs = [
        {'mean': 50, 'trend': 0, 'std': 2.0, 'name': '高水平平稳'},
        {'mean': 15, 'trend': 0, 'std': 1.5, 'name': '低水平平稳'},
        {'mean': 70, 'trend': 0.08, 'std': 2.5, 'name': '高水平上升'},
        {'mean': 5, 'trend': -0.05, 'std': 2.0, 'name': '低水平下降'},
        {'mean': 35, 'trend': 0, 'std': 3.0, 'name': '中水平高波动'},
        {'mean': 80, 'trend': 0, 'std': 1.5, 'name': '非常高水平'},
        {'mean': 10, 'trend': 0, 'std': 1.8, 'name': '低水平段'},
        {'mean': 45, 'trend': -0.1, 'std': 2.5, 'name': '中高水平快速下降'},
    ]
    
    # 生成每个段的数据
    for i in range(len(changepoints) - 1):
        start = changepoints[i]
        end = changepoints[i + 1]
        segment_length = end - start
        
        # 循环使用配置，确保有明显差异
        config = segment_configs[i % len(segment_configs)]
        mean = config['mean']
        trend = config['trend']
        base_std = config['std']
        
        # 生成段数据（添加趋势和噪声）
        # 减小噪声水平，使段内更平稳，变化点更明显
        segment_x = np.arange(segment_length)
        segment_data = (
            mean + 
            trend * segment_x + 
            np.random.normal(0, base_std, segment_length) * (noise_level * 0.3)  # 减小噪声
        )
        
        # 在每个段中添加明确的异常点
        if segment_length > 20:
            # 确保每个段有足够的异常点
            n_anomalies_in_segment = max(3, segment_length // 80)
            n_anomalies_in_segment = min(n_anomalies_in_segment, 12)
            
            anomaly_indices = np.random.choice(
                range(10, segment_length - 10),  # 避免在段边界添加异常点
                size=n_anomalies_in_segment,
                replace=False
            )
            
            for idx in anomaly_indices:
                # 使用段内的统计信息，而不是局部信息
                # 这样异常点相对于段内其他点更明显
                segment_mean = np.mean(segment_data)
                segment_std = np.std(segment_data) if len(segment_data) > 1 else base_std
                
                # 生成明显的异常值（至少6倍标准差，确保能被检测到）
                anomaly_magnitude = np.random.uniform(6, 12) * segment_std
                direction = 1 if np.random.random() > 0.5 else -1
                
                # 替换为异常值
                segment_data[idx] = segment_mean + direction * anomaly_magnitude
                true_anomalies.append(start + idx)
        
        data[start:end] = segment_data
    
    # 移除第一个变化点（起点）
    changepoints = changepoints[1:-1]
    
    # 额外添加一些全局极端异常点（跨段的极端值）
    n_global_anomalies = 8
    global_anomaly_positions = np.random.choice(
        range(100, n_samples - 100),
        size=n_global_anomalies,
        replace=False
    )
    
    for pos in global_anomaly_positions:
        if pos not in true_anomalies:  # 避免重复
            # 创建极端异常值（远离整体均值，确保在所有段中都很异常）
            overall_mean = np.mean(data)
            overall_std = np.std(data)
            extreme_value = overall_mean + np.random.choice([-1, 1]) * np.random.uniform(8, 15) * overall_std
            data[pos] = extreme_value
            true_anomalies.append(pos)
    
    true_anomalies = sorted(true_anomalies)
    
    return data, changepoints, true_anomalies


if __name__ == "__main__":
    print("=" * 70)
    print("生成PELT测试数据（包含明显的变化点和异常点）")
    print("=" * 70)
    
    data, true_cps, true_anomalies = generate_time_series_with_changepoints(
        n_samples=2000,
        n_changepoints=8,
        noise_level=0.3  # 降低噪声，使变化点更明显
    )
    
    # 生成时间戳
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(hours=i) for i in range(len(data))]
    
    # 创建标签
    labels = np.zeros(len(data), dtype=int)
    labels[true_anomalies] = 1
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': data,
        'label': labels,
    })
    
    # 保存
    df.to_csv("pelt_test_data.csv", index=False)
    print(f"✓ 数据集已保存到: pelt_test_data.csv")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 真实变化点数量: {len(true_cps)}")
    print(f"  - 真实变化点位置: {true_cps}")
    print(f"  - 真实异常点数量: {len(true_anomalies)}")
    print(f"  - 正常样本: {np.sum(labels == 0)}")
    print(f"  - 异常样本: {np.sum(labels == 1)}")
    print(f"  - 数据范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
    print(f"  - 数据均值: {np.mean(data):.2f}")
    print(f"  - 数据标准差: {np.std(data):.2f}")
    print("\n数据生成完成！")
    print(f"\n提示：数据包含 {len(true_cps)} 个变化点和 {len(true_anomalies)} 个异常点，")
    print(f"异常点均超过5倍标准差，应该能被PELT算法检测到。")
