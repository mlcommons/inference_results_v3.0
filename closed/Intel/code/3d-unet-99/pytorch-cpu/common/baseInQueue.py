"""
Base InQueue class
"""
from InputItem import InputItem

class baseInQueue:
    def __init__(self, mpQueue=None, batch_size=1, **kwargs):
        self.in_queue = mpQueue
        self.batch_size = batch_size

    def put(self, query_samples, receipt_time):
        """
        Receives query sample(s) from loadgen and processes the queries in batches if required/desired.
        Processed/batched queries are wrapped in 'InputItem' object, and then placed on the producer queue
        """
        num_samples = len(query_samples)
        if num_samples == 1:
            item = InputItem([query_samples[0].id], [query_samples[0].index], receipt_time=receipt_time)
            self.in_queue.put(item)

        else:
            #idx = [q.index for q in query_samples]
            #query_id = [q.id for q in query_samples]

            num_batches = num_samples // self.batch_size
            remainder = num_samples % self.batch_size
            batch = 0
            bidx = 0
            bs = self.batch_size
            while batch < num_batches:
                j = 0
                ids = []
                indexes = []
                while j < bs:
                    ids.append(query_samples[bidx].id)
                    indexes.append(query_samples[bidx].index)
                    bidx += 1
                    j += 1

                item = InputItem(ids, indexes, receipt_time=receipt_time)
                self.in_queue.put( item )
                batch += 1

            ids = []
            indexes = []
            while bidx < num_samples:
                ids.append(query_samples[bidx].id)
                indexes.append(query_samples[bidx].index)
                bidx += 1

                item = InputItem(ids, indexes, receipt_time=receipt_time)
                self.in_queue.put( item )

