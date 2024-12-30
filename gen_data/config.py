import os
import yaml
from omegaconf import OmegaConf
import pdb
from dataclasses import dataclass, field
from easydict import EasyDict
from typing import Any, Callable, Dict, List, Optional, Union

@dataclass
class GenDataConfig:
    # if os.environ['GEN_DATA_CONFIG_SETUP'] != '1':
    #     raise ValueError('Please check the data config setup')
    ROW_HEIGHT_SCALE_RATIO = 1.8  # Height scale of hw row compared to printing row. Scale biggest roi to this ratio of printing row height
    MIN_WORD_HEIGHT = 30  # a pasted hw word must has height greater than this
    DEFAULT_WORD_DIST = 4 # default x-axis distant between two words
    HIGH_TEXT_OFFSET_RATIO = 1/3  # if text has tall characters, move this roi down by this_ratio * roi.height
    BOX_CHECK_OVERLAP_THRESHOLD = 0.3 # use to check newly pasted box overlap with existing boxes
    MIN_WORD_TO_PASTE = 2
    MAX_ROW_TO_PASTE = 5


@dataclass
class Type1Config(GenDataConfig):
    pass

@dataclass
class Type2Config(GenDataConfig):
    pass

@dataclass
class Type3Config(GenDataConfig):
    pass


if __name__ == '__main__':
    print(Type1Config.XYZ)
    print(Type1Config.ROW_HEIGHT_SCALE_RATIO)
