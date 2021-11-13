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


#wgt = Baseline-tiered_imagenet-Conv64F-5-1-freeitter-Nov-10-2021-10-58-29
wgt = Baseline-tiered_imagenet-Conv64F-5-5-freeitter-Nov-10-2021-16-36-04

#ProtoNet-miniImageNet--ravi-resnet18-5-5-Nov-11-2021-00-40-51
#ProtoNet-miniImageNet--ravi-resnet18-5-1-Nov-10-2021-08-49-14

weight_root = './result_mosy/' + wgt

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
    print('\\\\\\\\\\\\\\\\\\\\' + weight_root + '>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
    print('\\\\\\\\\\\\\\\\\\\\' + args.data_root + '>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
    config = Config(os.path.join(weight_root, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, weight_root)
    test.test_loop()


