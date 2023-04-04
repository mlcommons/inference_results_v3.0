import time

class Item:
    """An item that we queue for processing by the thread pool."""
    def __init__(self, query_id, content_id, idx_offsets):
        self.query_id = query_id
        self.content_id = content_id
        self.idx_offsets = idx_offsets
        self.start = time.time()

class OItem:
    def __init__(self, presults, query_ids=None, array_ref=None, good=0, total=0, timing=0):
        self.good = good
        self.total = total
        self.timing = timing
        self.presults = presults
        self.query_ids = query_ids
        self.array_ref = array_ref

class kevin_OItem:
    def __init__(self, presults, query_ids=None, idx_offsets=None, good=0, total=0, timing=0):
        self.good = good
        self.total = total
        self.timing = timing
        self.presults = presults
        self.query_ids = query_ids
        self.idx_offsets = idx_offsets
        # self.start_loc = b0
        # self.end_loc = b1
