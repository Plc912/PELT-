# PELT时序分割异常检测工具

作者：庞力铖

邮箱：3522236586@qq.com

GitHub:

基于PELT (Pruned Exact Linear Time) 算法的时间序列变化点检测工具，通过将时序分割为平稳段，段内突变视为异常。

## 功能特点

- **PELT算法**：高效的精确线性时间变化点检测算法
- **时序分割**：将时间序列分割为多个平稳段
- **异常检测**：检测段内的突变点作为异常
- **动态规划**：使用动态规划找到最优变化点
- **MCP工具封装**：提供完整的MCP服务器接口
- **异步任务处理**：支持长时间运行的检测任务

## 项目结构

```
.
├── pelt/                      # 核心模块
│   ├── __init__.py            # 模块初始化
│   ├── pelt_detector.py       # PELT算法实现
│   └── preprocessor.py        # 数据预处理模块
├── pelt_mcp_server.py         # MCP服务器
├── requirements.txt           # 依赖包列表
└── README.md                  # 说明文档
```

## 环境要求

- Python 3.11-3.13
- pip (Python包管理器)

## 安装

```bash
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 启动MCP服务器

```bash
python pelt_mcp_server.py
```

服务器将在 `http://localhost:2263` 启动。

### 2. MCP工具说明

#### pelt_detect - 检测变化点和异常

使用PELT算法检测时间序列的变化点和异常。

**参数：**

- `filepath` (str, 必填): CSV文件路径
- `value_col` (str, 可选): 要分析的数值列名（None则自动选择第一个数值列）
- `timestamp_col` (str, 默认: "timestamp"): 时间戳列名
- `penalty` (float, 可选): 惩罚参数，控制检测敏感度（None则使用默认值2*log(10)）
- `min_segment_length` (int, 默认: 2): 最小段长度
- `anomaly_threshold` (float, 默认: 3.0): 异常阈值（标准差倍数）
- `skip_header` (int, 默认: 0): 跳过的头部行数
- `delimiter` (str, 默认: ","): 分隔符

**返回：**

```json
{
  "status": "queued",
  "task_id": "uuid",
  "type": "detect"
}
```

#### list_tasks - 列出所有任务

查询当前所有后台任务。

#### get_task - 获取任务详情

查询指定任务的详细状态、进度和结果。

**结果包含：**

- `n_changepoints`: 检测到的变化点数量
- `changepoints`: 变化点位置列表
- `n_segments`: 分段数量
- `segments`: 分段列表 [(start, end), ...]
- `n_anomalies`: 异常点数量
- `anomaly_indices`: 异常点位置列表
- `anomaly_scores`: 异常分数列表

## 算法原理

PELT (Pruned Exact Linear Time) 是一种高效的变化点检测算法：

1. **动态规划**：使用动态规划找到使总成本最小的变化点集合
2. **剪枝优化**：通过剪枝策略减少计算量，实现线性时间复杂度
3. **成本函数**：基于高斯分布的负对数似然作为段成本
4. **惩罚机制**：通过penalty参数平衡检测敏感度和过度分割

### 工作流程

1. **分割序列**：将时间序列分割为多个平稳段
2. **段内检测**：在每个平稳段内使用统计方法（Z-score）检测异常
3. **结果输出**：返回变化点位置、分段信息和异常点位置

## 应用场景

- 传感器数据变化点检测
- 股票价格趋势变化检测
- 系统性能监控中的异常检测
- 网络流量模式变化检测
- 工业过程控制中的故障检测

## 技术细节

- **算法复杂度**: O(n)，线性时间复杂度
- **成本函数**: 基于高斯分布的负对数似然
- **异常检测**: 基于Z-score的统计方法
- **默认penalty**: 2*log(10) ≈ 4.6
- **默认异常阈值**: 3.0（3倍标准差）

## 使用示例

### Python示例

```python
from pelt import PELTDetector, Preprocessor

# 加载数据
preprocessor = Preprocessor()
data, timestamps, column_name = preprocessor.load_csv("pelt_test_data.csv")

# 创建检测器
detector = PELTDetector(
    penalty=2.0 * np.log(10),
    min_segment_length=2,
)

# 检测变化点
changepoints, segments = detector.detect(data, return_segments=True)

# 检测异常
anomaly_indices, anomaly_scores = detector.detect_anomalies(
    data,
    anomaly_threshold=3.0,
)

print(f"检测到 {len(changepoints)} 个变化点")
print(f"检测到 {len(anomaly_indices)} 个异常点")
```

### MCP客户端示例

```python
# 检测变化点和异常
response = mcp_client.call_tool(
    "pelt_detect",
    filepath="pelt_test_data.csv",
    penalty=None,  # 使用默认值
    anomaly_threshold=3.0,
)

task_id = response["task_id"]

# 查询结果
task = mcp_client.call_tool("get_task", task_id=task_id)
result = task["result"]
print(f"变化点: {result['changepoints']}")
print(f"异常点: {result['anomaly_indices']}")
```

## 参数调优

### 调整 penalty参数

- **较小值 (2*log(n))**: 更敏感，检测更多变化点
- **较大值 (2*log(1000))**: 更不敏感，只检测显著变化
- **默认值 (2*log(10))**: 平衡敏感度和稳定性

### anomaly_threshold参数

- **较小值 (2.0)**: 检测更多异常点（可能有误报）
- **较大值 (4.0)**: 只检测极端异常（可能漏报）
- **默认值 (3.0)**: 3-sigma规则，平衡准确性和召回率

## 注意事项

1. 数据应为一维时间序列
2. 建议数据量不少于100个样本
3. penalty参数影响变化点检测敏感度
4. min_segment_length应至少为2
5. 异常检测基于统计方法，适用于正态分布的数据

## 许可证

MIT License

## 作者

PELT时序分割异常检测工具开发团队
