from typing import Iterable, Dict, List

_rn50_multi_instance_bs8_raw_cfg = '64,64,1,1,28,-1,-1,5,64,64,1,28,8,-1,-1,2,32,64,1,-1,-1,28,-1,1,32,64,1,2,2,-1,-1,4,64,32,1,14,8,-1,-1,0,16,64,1,-1,-1,56,-1,3,32,64,1,2,8,-1,-1,4,64,32,1,8,2,-1,-1,4,16,64,1,-1,-1,28,-1,2,32,64,1,2,14,-1,-1,4,64,64,1,4,4,-1,1,2,64,64,1,8,56,-1,-1,1,128,64,1,-1,-1,797,-1,1,64,128,1,2,4,-1,-1,1,32,128,1,4,7,-1,-1,4,128,128,1,-1,-1,419,-1,1,128,64,1,1,28,-1,-1,1,64,32,1,14,14,-1,-1,2,32,64,1,-1,-1,419,-1,1,32,128,1,4,2,-1,-1,4,32,64,1,2,2,-1,-1,3,32,64,1,-1,-1,419,-1,2,64,64,1,2,4,-1,-1,4,32,64,1,14,14,-1,1,1,32,64,1,2,14,-1,-1,2,64,64,1,-1,-1,404,-1,2,32,256,1,2,1,-1,-1,4,256,64,1,2,1,-1,-1,1,32,64,1,-1,-1,222,-1,2,64,256,1,2,1,-1,-1,0,64,512,1,14,2,-1,-1,4,16,128,1,-1,-1,222,-1,3,512,256,1,2,2,-1,-1,3,64,64,1,14,1,-1,-1,2,16,64,1,-1,-1,222,-1,2,64,128,1,14,14,-1,-1,4,32,128,1,14,14,-1,-1,4,32,128,1,-1,-1,222,-1,0,128,64,1,1,7,-1,-1,2,128,256,1,2,2,-1,-1,5,128,128,1,-1,-1,222,-1,1,256,256,1,2,14,-1,-1,5,32,1024,1,7,7,-1,1,3,256,1024,1,7,7,-1,-1,2,64,256,1,7,7,-1,-1,1,32,64,1,7,7,-1,-1,3,64,64,1,7,7,-1,-1,3,128,64,1,-1,-1,61,-1,3,64,512,1,1,7,-1,-1,5,128,64,1,7,1,-1,-1,5,128,512,1,-1,-1,61,-1,3,64,256,1,7,1,-1,-1,1'
_rn50_multi_instance_bs256_raw_cfg = '64,64,1,1,56,-1,-1,0,64,64,1,2,56,-1,-1,1,64,64,1,-1,-1,56,-1,0,64,64,1,1,56,-1,-1,1,64,64,1,1,56,-1,-1,0,64,64,1,-1,-1,56,-1,1,64,64,1,1,56,-1,-1,1,64,64,1,1,56,-1,-1,0,64,64,1,-1,-1,56,-1,0,64,64,1,1,56,-1,-1,0,64,64,1,2,28,-1,1,0,64,64,1,1,28,-1,-1,0,64,64,1,1,28,-1,-1,1,64,64,1,2,28,-1,-1,1,64,64,1,1,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,1,28,-1,-1,0,64,64,1,14,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,4,28,-1,-1,0,64,64,1,2,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,2,28,-1,-1,0,64,64,1,2,14,-1,1,1,64,64,1,2,28,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,1,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,1,64,64,1,2,14,-1,-1,1,512,64,1,7,7,-1,1,1,512,64,1,2,14,-1,-1,1,256,64,1,7,7,-1,-1,0,512,512,1,7,7,-1,-1,0,64,512,1,7,7,-1,-1,0,128,64,1,-1,-1,61,-1,0,512,512,1,7,7,-1,-1,1,256,512,1,7,7,-1,-1,0,64,512,1,-1,-1,61,-1,0,512,64,1,7,7,-1,-1,0'
unified_raw_cfg = '64,64,1,1,56,-1,-1,0,64,64,1,2,56,-1,-1,1,64,64,1,-1,-1,56,-1,0,64,64,1,1,56,-1,-1,1,64,64,1,1,56,-1,-1,0,64,64,1,-1,-1,56,-1,1,64,64,1,1,56,-1,-1,1,64,64,1,1,56,-1,-1,0,64,64,1,-1,-1,56,-1,0,64,64,1,1,56,-1,-1,0,64,64,1,2,28,-1,1,0,64,64,1,1,28,-1,-1,0,64,64,1,1,28,-1,-1,1,64,64,1,2,28,-1,-1,1,64,64,1,1,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,1,28,-1,-1,0,64,64,1,14,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,4,28,-1,-1,0,64,64,1,2,28,-1,-1,0,64,64,1,-1,-1,419,-1,1,64,64,1,2,28,-1,-1,0,64,64,1,2,14,-1,1,1,64,64,1,2,28,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,1,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,2,14,-1,-1,0,64,64,1,-1,-1,222,-1,1,64,64,1,2,14,-1,-1,1,512,64,1,7,7,-1,1,1,512,64,1,2,14,-1,-1,1,256,64,1,7,7,-1,-1,0,512,512,1,7,7,-1,-1,0,64,512,1,7,7,-1,-1,0,128,64,1,-1,-1,61,-1,0,512,512,1,7,7,-1,-1,1,256,512,1,7,7,-1,-1,0,64,512,1,-1,-1,61,-1,0,512,64,1,7,7,-1,-1,0'


class ConvFwdConfig:
    number_of_config_elements = 8

    def __init__(self, k_block: int, c_block: int, tile_d: int, tile_p: int, tile_q: int, tile_os: int,
                 pack_input: int, loop_sched: int):
        self.k_block = k_block
        self.c_block = c_block
        self.tile_d = tile_d
        self.tile_p = tile_p
        self.tile_q = tile_q
        self.tile_os = tile_os
        self.pack_input = pack_input
        self.loop_sched = loop_sched

    def to_dict(self) -> Dict[str, int]:
        return {
            'K_block': self.k_block,
            'C_block': self.c_block,
            'tile_d': self.tile_d,
            'tile_p': self.tile_p,
            'tile_q': self.tile_q,
            'tile_os': self.tile_os,
            'pack_input': self.pack_input,
            'loop_sched': self.loop_sched,
        }

    def to_list(self) -> List[int]:
        return [
            self.k_block,
            self.c_block,
            self.tile_d,
            self.tile_p,
            self.tile_q,
            self.tile_os,
            self.pack_input,
            self.loop_sched,
        ]

def get_unified_cfgs() -> Iterable[ConvFwdConfig]:
    return _get_cfgs(unified_raw_cfg)


def get_rn50_multi_instance_bs8_cfgs() -> Iterable[ConvFwdConfig]:
    return _get_cfgs(_rn50_multi_instance_bs8_raw_cfg)


def get_rn50_multi_instance_bs256_cfgs() -> Iterable[ConvFwdConfig]:
    return _get_cfgs(_rn50_multi_instance_bs256_raw_cfg)


def _get_cfgs(raw_cfg: str) -> Iterable[ConvFwdConfig]:
    split_cfgs = [int(cfg) for cfg in raw_cfg.split(',')]
    length = len(split_cfgs)
    elements = ConvFwdConfig.number_of_config_elements
    if length % elements != 0:
        raise ValueError('Illegal config, length % elements != 0: {}'.format(raw_cfg))
    for i in range(0, length, elements):
        yield ConvFwdConfig(*split_cfgs[i:i+elements])