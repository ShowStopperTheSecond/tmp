# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import numpy as np
from PIL import Image

from .dataset import Dataset
from .pair_dataset import PairDataset, StillPairDataset


class SARImages (Dataset):
    """ Loads all images from the Aachen Day-Night dataset 
    """
    def __init__(self, select='agri urban', root='data/sar'):
        Dataset.__init__(self)
        self.root = root
        self.img_dir = 's1'
        self.select = set(select.split())
        assert self.select, 'Nothing was selected'
        
        self.imgs = []
        root = os.path.join(root, self.img_dir)
        for dirpath, _, filenames in os.walk(root):
            r = dirpath[len(root)+1:]
            if not(self.select & set(r.split('/'))): continue
            self.imgs += [os.path.join(r,f) for f in filenames if f.endswith('.png')]
        
        self.nimg = len(self.imgs)
        assert self.nimg, 'Empty SAR dataset'

    def get_key(self, idx):
        return self.imgs[idx]



class SARImages_DB (SARImages):
    """ Only database (db) images.
    """
    def __init__(self, **kw):
        SARImages.__init__(self, select='db', **kw)
        self.db_image_idxs = {self.get_tag(i) : i for i,f in enumerate(self.imgs)}
    
    def get_tag(self, idx): 
        # returns image tag == img number (name)
        return os.path.split( self.imgs[idx][:-4] )[1]



if __name__ == '__main__':
    print(aachen_db_images)
    print(aachen_style_transfer_pairs)
    print(aachen_flow_pairs)
    pdb.set_trace()
