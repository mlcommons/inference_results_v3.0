"""
abstract baseBackend class
"""
class baseBackend():
    def __init__(self):
        pass
    def load_model(self):
        raise NotImplementedError("baseBackend:load")

    def predict(self):
        raise NotImplementedError("baseBackend:predict")