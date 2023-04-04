class InputItem:
    def __init__(self, id_list, index_list, data=None, label=None, receipt_time=0):
        self.query_id_list = id_list    # e.g. [367122711312] 
        self.sample_index_list = index_list   #query data id list e.g. [17]
        self.data = data
        self.label = label
        self.receipt_time = receipt_time
