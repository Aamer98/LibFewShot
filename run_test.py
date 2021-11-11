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
parser.add_argument('--weight_root', default='./results/ProtoNet-miniImageNet--ravi-Conv64F-5-1-Nov-10-2021-05-19-50', help = 'path to weight')


args = parser.parse_args()
VAR_DICT = {
    "data_root": args.data_root,
    "test_epoch": 5,
    "device_ids": "0",
    "n_gpu": 2,
    "test_episode": 2000,
    "episode_size": 1,
    "test_way": 5}


if __name__ == "__main__":
    print('\\\\\\\\\\\\\\\\\\\\' + args.weight_root + '>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
    print('\\\\\\\\\\\\\\\\\\\\' + args.data_root + '>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
    config = Config(os.path.join(args.weight_root, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, args.weight_root)
    test.test_loop()


