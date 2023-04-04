import mlperf_loadgen as lg
import torch


class RNNTQSL:
    def __init__(self, dataset_dir, perf_count=None):
        self.dataset = list(torch.load(dataset_dir).values())  # {"x":[], "x_lens":[]}
        self.count = len(self.dataset[0])
        perf_count = self.count if perf_count is None else perf_count
        self.sample_id_to_sample = {}
        self.qsl = lg.ConstructQSL(
            self.count, perf_count, self.load_query_samples, self.unload_query_samples
        )
        self.load_query_samples(range(self.count))
        print(f"Number of samples: {self.count}")

    def load_query_samples(self, sample_list):
        for sample_id in sample_list:
            self.sample_id_to_sample[sample_id] = self._load_sample(sample_id)

    def unload_query_samples(self, sample_list):
        for sample_id in sample_list:
            del self.sample_id_to_sample[sample_id]

    def _load_sample(self, index):
        sample = (self.dataset[0][index], self.dataset[1][index])
        return sample

    def __getitem__(self, index):
        return self.sample_id_to_sample[index]

    def __del__(self):
        lg.DestroyQSL(self.qsl)
        print("Finished destroying QSL.")
