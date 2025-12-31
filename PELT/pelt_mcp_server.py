"""
PELT时序分割异常检测MCP服务器
通过变化点检测将时序分割为平稳段，段内突变视为异常
"""
from typing import Optional, Dict, Any
import os
import threading
import uuid
import datetime
import traceback

import numpy as np
from fastmcp import FastMCP

from pelt import PELTDetector, Preprocessor

mcp = FastMCP("PELT Anomaly Detection")

# ---- Background task infrastructure ----
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
MAX_CONCURRENT = int(os.getenv("PELT_MAX_CONCURRENT", "2"))
TASKS_SEM = threading.Semaphore(MAX_CONCURRENT)

# China Standard Time (UTC+08:00)
TZ_CN = datetime.timezone(datetime.timedelta(hours=8))

def _now_iso() -> str:
    return datetime.datetime.now(TZ_CN).isoformat()


def _create_task(task_type: str, params: Dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": task_type,
        "params": params,
        "status": "queued",
        "progress": 0.0,
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "traceback": None,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
    return task_id


def _set_task(task_id: str, **updates):
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(**updates)


def _get_task(task_id: str) -> Dict[str, Any]:
    with TASKS_LOCK:
        return dict(TASKS.get(task_id, {}))


def _list_tasks() -> list:
    with TASKS_LOCK:
        return [dict(t) for t in TASKS.values()]


def _detect_worker(task_id: str, params: Dict[str, Any]):
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            preprocessor = Preprocessor()
            filepath = params.get("filepath")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            _set_task(task_id, progress=0.2)
            
            # 加载数据
            data, timestamps, column_name = preprocessor.load_csv(
                filepath=filepath,
                value_col=params.get("value_col"),
                timestamp_col=params.get("timestamp_col", "timestamp"),
                skip_header=params.get("skip_header", 0),
                delimiter=params.get("delimiter", ","),
            )
            
            _set_task(task_id, progress=0.4)
            
            # 创建检测器
            penalty = params.get("penalty")
            if penalty is None:
                penalty = 2.0 * np.log(10)
            
            detector = PELTDetector(
                penalty=penalty,
                min_segment_length=params.get("min_segment_length", 2),
            )
            
            _set_task(task_id, progress=0.5)
            
            # 检测变化点
            changepoints, segments = detector.detect(
                data,
                return_segments=True,
            )
            
            _set_task(task_id, progress=0.7)
            
            # 检测异常点
            anomaly_threshold = params.get("anomaly_threshold", 3.0)
            anomaly_indices, anomaly_scores = detector.detect_anomalies(
                data,
                anomaly_threshold=anomaly_threshold,
            )
            
            _set_task(task_id, progress=0.9)
            
            # 构建结果
            result = {
                "status": "ok",
                "n_samples": int(len(data)),
                "n_changepoints": int(len(changepoints)),
                "changepoints": [int(cp) for cp in changepoints],
                "n_segments": int(len(segments) if segments else 0),
                "segments": [(int(s), int(e)) for s, e in segments] if segments else [],
                "n_anomalies": int(len(anomaly_indices)),
                "anomaly_indices": [int(idx) for idx in anomaly_indices],
                "anomaly_scores": [float(score) for score in anomaly_scores],
                "column_name": column_name,
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result=result)
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(),
                 error=str(ex), traceback=traceback.format_exc())


def _start_background(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


@mcp.tool()
def pelt_detect(
    filepath: str,
    value_col: Optional[str] = None,
    timestamp_col: str = "timestamp",
    penalty: Optional[float] = None,
    min_segment_length: int = 2,
    anomaly_threshold: float = 3.0,
    skip_header: int = 0,
    delimiter: str = ",",
) -> Dict[str, Any]:
    """
    使用PELT算法检测时间序列的变化点和异常
    
    Args:
        filepath: CSV文件路径
        value_col: 要分析的数值列名（None则自动选择第一个数值列）
        timestamp_col: 时间戳列名
        penalty: 惩罚参数，控制检测敏感度（None则使用默认值2*log(10)）
        min_segment_length: 最小段长度
        anomaly_threshold: 异常阈值（标准差倍数）
        skip_header: 跳过的头部行数
        delimiter: 分隔符
        
    Returns:
        任务信息字典
    """
    params = {
        "filepath": filepath,
        "value_col": value_col,
        "timestamp_col": timestamp_col,
        "penalty": penalty,
        "min_segment_length": min_segment_length,
        "anomaly_threshold": anomaly_threshold,
        "skip_header": skip_header,
        "delimiter": delimiter,
    }
    
    task_id = _create_task("detect", params)
    _start_background(_detect_worker, task_id, params)
    
    return {"status": "queued", "task_id": task_id, "type": "detect"}


@mcp.tool()
def list_tasks() -> Dict[str, Any]:
    """列出所有后台任务"""
    tasks = _list_tasks()
    return {"count": len(tasks), "tasks": tasks}


@mcp.tool()
def get_task(task_id: str) -> Dict[str, Any]:
    """获取指定任务的详细信息"""
    return _get_task(task_id)


if __name__ == "__main__":
    mcp.run(transport="sse", port=2263)

