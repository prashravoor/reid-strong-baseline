import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Amur(BaseImageDataset):
    """
       Amur Tiger Dataset
       Reference:
       S. Li et al. "Amur Tiger Re-identification in the Wild", 2019.

       URL: https://cvwc2019.github.io/challenge.html

       Dataset statistics:
       # Uses only the plain-reid-train dataset, of which 20% of the ids
       # and corresponding images are used for the test set

       # identities: 85
       # images: 812 (train) + 241 (query) + 499 (gallery)
       # cameras: 1
       """

    dataset_dir = 'amur'

    def __init__(self, root='../', verbose=True, **kwargs):
        super(Amur, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Amur Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d+)_(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid,_ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 0 <= camid <= 20
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

