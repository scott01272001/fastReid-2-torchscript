import cv2
import torch
import argparse
import sys
import os
sys.path.append('.')
from fastreid.modeling.meta_arch import build_model
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger

setup_logger(name="fastreid")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--image",
        help="path to the image used to feed to the model",
    )

    parser.add_argument(
        "--output",
        default='torchscript_model',
        help='path to save model file'
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_save_directory():
    cwd = os.getcwd()
    cwd = os.path.join(cwd, "ts_model")
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    return cwd

def load_feed_image(cfg, image_path):
    original_image = cv2.imread(image_path)
    original_image = original_image[:, :, ::-1]
    image = cv2.resize(original_image, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
    return image

def load_model(cfg):
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    return model

def check_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    
    model = load_model(cfg)
    feed_img = load_feed_image(cfg, args.image)

    traced_script_module = torch.jit.trace(model, feed_img.to(model.device))

    head, tail = os.path.split(cfg.MODEL.WEIGHTS)
    tail = tail.replace(".pth", "_ts.pt")
    save_path = os.path.join(args.output, tail)
    check_dir_exist(args.output)

    traced_script_module.save(save_path)
    print("Save ts model to " + save_path)
