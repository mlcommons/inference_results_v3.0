
class OutputItem:
    def __init__(self, query_id_list, result, array_type_code='B'):
        self.query_id_list = query_id_list
        self.result = result
        self.array_type_code = array_type_code
        self.receipt_time = None
        self.outqueue_time = None

    def set_receipt_time(self, receipt_time):
        self.receipt_time = receipt_time

    def set_outqueued_time(self, outqueue_time):
        self.outqueue_time = outqueue_time

    

