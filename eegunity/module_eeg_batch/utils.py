import numpy as np
# Convert all fields in info to JSON-compatible types
def __convert_to_serializable__(value):
    # 检查是否为时间对象，使用 isoformat 转换
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    # 检查是否为 ndarray 类型，转换为列表
    elif isinstance(value, np.ndarray):
        return value.tolist()
    # 检查是否为其他非序列化类型，直接转换为字符串
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    # 默认返回原值
    return value