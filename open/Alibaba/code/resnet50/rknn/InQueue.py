import logging

import numpy as np
from InputItem import InputItem
from baseInQueue import baseInQueue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ENQUEUE")


class InQueue(baseInQueue):
    def __init__(self, mpQueue=None, batch_size=1, min_query_count=1, **kwargs):
        """
        Will be instantiated with keyword-val dictionary
        Initializer should have named inputs.
        As a generic rull, init should have named input 'mpQueue'
        which is a multiprocessing module's JoinableQueue instance
        """
        # TODO: We may chose to pass mpQueue as list even if not bucketing
        self.in_queue = mpQueue if isinstance(mpQueue, list) else [mpQueue] #[mpQueue: mp.JoinableQueue()]
        self.batch_size = batch_size
        self.curr_query_count = 0
        self.qid_list = []
        self.qidx_list = [] #query data index

    def put(self, query_samples, receipt_time=0):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        query_samples: list of mlperf_loadgen.QuerySample
        """
        num_samples = len(query_samples)
        # TODO: Remove all logging in here
        # log.info("Adding {} samples to queue".format(num_samples))
        if num_samples == 1:
            self.curr_query_count += 1
            self.qid_list.append(query_samples[0].id)
            self.qidx_list.append(query_samples[0].index)
            if self.curr_query_count == self.batch_size:
                item = InputItem(self.qid_list, self.qidx_list, receipt_time=receipt_time)
                w_idx = np.random.randint(0, len(self.in_queue))
                self.in_queue[w_idx].put(item)
                self.curr_query_count = 0
                self.qid_list = []
                self.qidx_list = []

        else:
            idx = [q.index for q in query_samples]
            query_id = [q.id for q in query_samples]

            num_batches = num_samples // self.batch_size
            remainder = num_samples % self.batch_size
            batch = 0
            bidx = 0
            bs = self.batch_size
            while batch < num_batches:
                ids = query_id[bidx:bidx + bs]
                indexes = idx[bidx:bidx + bs]
                item = InputItem(ids, indexes, receipt_time=receipt_time)  # , data, label)

                w_idx = np.random.randint(0, len(self.in_queue))
                self.in_queue[w_idx].put(item)
                batch += 1
                bidx += bs

            if remainder > 0:
                ids = query_id[bidx:]
                indexes = idx[bidx:]
                item = InputItem(ids, indexes)  # , data, label)
                self.in_queue[0].put(item)

    def put_last_batch(self, receipt_time=0):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        """
        if self.qid_list:
            item = InputItem(self.qid_list, self.qidx_list, receipt_time=receipt_time)
            self.in_queue[0].put(item)
