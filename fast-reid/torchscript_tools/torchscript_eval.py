#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

import torch
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from custom_defaults import DefaultTrainer, default_argument_parser, default_setup
from fastreid.engine import launch

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = torch.jit.load(cfg.MODEL.WEIGHTS, map_location=torch.device(cfg.MODEL.DEVICE))

    res = DefaultTrainer.test(cfg, model)
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
