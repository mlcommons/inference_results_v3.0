class InputData:
    def __init__(self, data=None, data_shape=None, sequence_len=None):
        self.data = data #numpy, (nchw)
        self.input_shape = data_shape #chw
        self.sequence_len = sequence_len
