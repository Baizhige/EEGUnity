class UDatasetSharedAttributes:
    def __init__(self):
        self._shared_attr = {}  # 使用实例变量来存储共享属性

    def get_shared_attr(self):
        return self._shared_attr

    def set_shared_attr(self, value):
        self._shared_attr.update(value)