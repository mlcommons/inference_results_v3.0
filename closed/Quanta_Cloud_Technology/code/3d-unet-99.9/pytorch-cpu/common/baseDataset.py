"""
abstract baseDataset class
"""
class baseDataset():
    def __init__(self):
        pass
    def get_warmup_samples(self):
        raise NotImplementedError("baseDataset:get_warmup_samples")
    def get_samples(self, sample_index_list=[]):
        raise NotImplementedError("baseDataset:get_samples")
    def post_process(self, query_ids, results):
        raise NotImplementedError("baseDataset:post_process")
    def load_query_samples(self, sample_index_list):
        pass
    def unload_query_samples(self, sample_index_list):
        pass
    def load_data(self):
        pass