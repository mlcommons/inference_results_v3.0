from rknnlite.api import RKNNLite

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RKNN_model_container")

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn_lite = RKNNLite()

        # Direct Load RKNN Model
        log.info('--> Load RKNN model')
        ret = rknn_lite.load_rknn(model_path)
        if ret != 0:
            log.info('Load RKNN model failed')
            exit(ret)
        log.info('done')

        log.info('--> Init runtime environment')
        self.core_mask = RKNNLite.NPU_CORE_AUTO
        #self.core_mask = RKNNLite.NPU_CORE_0_1_2
        ret = rknn_lite.init_runtime(core_mask=self.core_mask)

        if ret != 0:
            log.info('Init runtime environment failed')
            exit(ret)
        log.info('done')
        
        self.rknn_lite = rknn_lite 

    def run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]
        result = self.rknn_lite.inference(inputs=inputs)
        return result
    
    def release(self):
        self.rknn_lite.release()
