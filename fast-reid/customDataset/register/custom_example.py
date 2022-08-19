
import glob
import os.path as osp
import re

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CustomDataset(ImageDataset):

    """Custom dataset

    This dataset register class is used for training on custom dataset.

    The dataset hierarchy is mandatory as below:
        -custom-data
            |-bounding_box_test
            |-bounding_box_train
            |-query

    Naming rule of the data:
        example: 0002_c1_f0044158.jpg
            - 0002: label
            - c1: camera number
            - f0044158: the 0044158th frame in the camera c1

    The parameter 'camera_count' means the number of the camera in your dataset, 
    in default, the parameter value is set to 8, please specify the actual value according to your dataset.

    To make the register effactive, please import the class in train_net.py.

    """
    camera_count = 8
    dataset_dir = 'custom-data'
    dataset_name = "custom"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(CustomDataset, self).__init__(train, query, gallery, **kwargs)
    
    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= self.camera_count
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
        
