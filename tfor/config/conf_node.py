import itertools
from typing import AnyStr, Dict, Optional, List

from yacs.config import CfgNode as CNode


class ConfNode(CNode):
    def __init__(self,
                 init_dict: Optional[Dict] = None,
                 key_list: Optional[List[AnyStr]] = None,
                 new_allowed: bool = False):
        super(ConfNode, self).__init__(init_dict=init_dict, key_list=key_list, new_allowed=new_allowed)

    def unfreeze(self):
        self.defrost()

    def merge_from_file(self, cfg_filename: AnyStr):
        """Load a yaml config file and merge it this CfgNode."""
        _extra_cfgs = []
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
            for k, v in cfg.items():
                if k.startswith('__') and isinstance(v, str):
                    _extra_cfgs.append(ConfNode(init_dict={k.strip('_'): self.load_cfg(f)}))
        self.merge_from_other_cfg(cfg)
        for _ec in _extra_cfgs:
            self.merge_from_other_cfg(_ec)

    def merge_from_dict(self, conf_dic: Dict):
        cfg_list = list(itertools.chain(*conf_dic.items()))
        self.merge_from_list(cfg_list)
