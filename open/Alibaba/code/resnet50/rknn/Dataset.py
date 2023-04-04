import logging
import os
import re
import sys

import math
import numpy as np
from InputData import InputData
from OutputItem import OutputItem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")


class Dataset():
    def __init__(self, dataset_param, mlperf_param):
        self.image_filenames = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.label_list_inmemory = {}
        self.count = mlperf_param["total_sample_count"]
        self.batch_size = dataset_param["batch_size"]
        self.image_size = dataset_param["image_size"]
        self.precision = dataset_param["precision"]
        self.layout = dataset_param["layout"]
        self.image_list = dataset_param["image_list"]
        self.dataset = dataset_param["dataset"]
        self.scale = dataset_param["scale"]
        self.offset = dataset_param["offset"]

        self.mode = mlperf_param['mode']

        if self.image_list is None:
            self.image_list = os.path.join(self.dataset, "val_map.txt")

        if not os.path.isfile(self.image_list):
            log.error("image list not found: {}".format(self.image_list))
            sys.exit(1)

        not_found = 0
        with open(self.image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(self.dataset, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                self.image_filenames.append(src)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_filenames) >= self.count:
                    break

        self.count = min(self.count, len(self.image_filenames))
        if self.mode == 'performance':
            self.load_dataset_into_memory()

    def load_query_samples(self, sample_index_list):
        """
        Called by loadgen to load samples before sending queries to sut.
        Ideally complementary to load_dataset. If using this to load samples by loadgen,
        the samples are not necessarily available across processes
        - Needs to figure out if possible to work this out
        """
        pass

    def getCount(self):
        return self.count #self.counts

    def load_dataset_into_memory(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """
        log.info("Loading dataset into memory")
        for index in range(self.count):
            src = self.image_filenames[index]
            processed = self.pre_process(src)
            self.image_list_inmemory[index] = processed
            self.label_list_inmemory[index] = self.label_list[index]

    def load_dataset(self):
        """
        Responsible for loading all available dataset into memory.
        Ideally complementary to 'load_query_samples
        """

    def unload_query_samples(self, sample_list):
        """
        Workload dependent. But typically not implemented if load_query_samples is not implemented
        """
        log.info("Called to unload data")
        pass

    def obj_unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.image_list_inmemory:
                    del self.image_list_inmemory[sample]
                    del self.label_list_inmemory[sample]
        else:
            self.image_list_inmemory = {}
            self.label_list_inmemory = {}

    def get_samples(self, sample_index_list=[]):
        """
        Fetches and returns pre-processed data at requested 'sample_index_list'
        """
        outData = []
        for idx in sample_index_list:
            if self.mode == 'performance':
                data = self.image_list_inmemory[idx]
            elif self.mode == 'accuracy':
                data = self.sample_by_idx(idx)
            
            outData.append(data)

        data = np.asarray(outData)
        return InputData(data=data, data_shape=data.shape)

    def get_warmup_samples(self):
        """
        Fetches and returns pre-processed data for warmup
        """
        import random
        num_samples = self.batch_size
        warmup_samples = []
        outData = []
        if len(self.image_list_inmemory) < num_samples:
            self.load_query_samples(list(range(num_samples)))
        if self.mode == 'performance':
            sample_ids = random.choices(list(self.image_list_inmemory.keys()), k=num_samples)
        elif self.mode == 'accuracy':
            sample_ids = random.choices(list(range(len(self.image_filenames))), k=num_samples)

        for idx in sample_ids:
            if self.mode == 'performance': 
                data_item = self.image_list_inmemory[idx]
            elif self.mode == 'accuracy':
                data_item = self.sample_by_idx(idx)

            outData.append(data_item)
        data_item = np.asarray(outData)
        item = InputData(data=data_item, data_shape=data_item.shape)
        warmup_samples.append(item)

        return warmup_samples

    def pre_process(self, file=''):
        """
        Pre-processes a given input/image
        """
        #log.info(f'pre_process {file} 0')
        import cv2
        from PIL import Image
        import torchvision.transforms.functional as F
        from torchvision import transforms

        pid = os.getpid()

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = F.resize(img, int(math.ceil(self.image_size / 0.875)), Image.BILINEAR)
        img = F.center_crop(img, self.image_size)

        img = transforms.ToTensor()(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        if self.precision == "int8":
            img = img / self.scale - self.offset
        img = np.asarray(img, dtype=self.precision)
        if self.layout == "NHWC":
            img = np.transpose(img, [1, 2, 0])
        
        return img

    def post_process(self, query_ids, sample_index_list, results):
        """
        Post-processor that accepts loadgens query ids and corresponding inference output.
        post_process should return and OutputItem object which has two attributes:
        OutputItem.query_id_list
        OutputItem.results
        """
        processed_results = []
        results = np.argmax(results, axis=1)
        n = results.shape[0]
        for idx in range(n):
            result = results[idx]
            processed_results.append([result])
        return OutputItem(query_ids, processed_results, array_type_code='q')

    def sample_by_idx(self, index):
        return self.pre_process(self.image_filenames[index])
