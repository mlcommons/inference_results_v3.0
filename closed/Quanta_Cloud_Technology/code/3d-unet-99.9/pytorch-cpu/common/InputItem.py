
class InputItem:
    def __init__(self, id_list, index_list, data=None, label=None, receipt_time=0):
        self.query_id_list = id_list
        self.sample_index_list = index_list
        self.data = data
        self.label = label
        self.receipt_time = receipt_time
