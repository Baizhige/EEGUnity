class UDatasetSharedAttributes:
    def __init__(self):
        self._shared_attr = {}  # Use instance variables to store shared attributes
    def get_shared_attr(self):
        return self._shared_attr

    def set_shared_attr(self, value):
        self._shared_attr.update(value)