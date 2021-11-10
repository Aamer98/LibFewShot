# -*- coding: utf-8 -*-
import argparse
from core import Trainer
from core import Test
from core.config import Config
import os
import sys

sys.dont_write_bytecode = True


import argparse
from core import Trainer
from core import Test
from core.config import Config
import os
import sys

parser = argparse.ArgumentParser(description='LibFewShot Training')

parser.add_argument('--data_root', default='./dataset/miniImageNet--ravi',  help='path to dataset')


Path_list = ['./results/ProtoNet-miniImageNet--ravi-Conv64F-5-1-Nov-08-2021-09-05-05',
  './results/ProtoNet-miniImageNet--ravi-Conv64F-5-20-Nov-08-2021-09-05-01', './results/ProtoNet-miniImageNet--ravi-Conv64F-5-5-Nov-08-2021-12-11-56']

args = parser.parse_args()
VAR_DICT = {
    "data_root": args.data_root,
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 2000,
    "episode_size": 1,
    "test_way": 5}


if __name__ == "__main__":
    for PATH in Path_list:
        print('\\\\\\\\\\\\\\\\\\\\' + PATH + '>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
        config = Config(os.path.join(PATH, "config.yaml"),
                    VAR_DICT).get_config_dict()
        test = Test(config, PATH)
        test.test_loop()


