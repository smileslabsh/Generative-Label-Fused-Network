# -----------------------------------------------------------
# Generative Label Fused Network implementation based on
# Position Focused Attention Network (PFAN) and Stacked Cross Attention Network (SCAN)
# the code of PFAN: https://github.com/HaoYang0123/Position-Focused-Attention-Network
# the code of SCAN: https://github.com/kuanghuei/SCAN
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import parts
import dgl

split_size = '_16_15'


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
        self.data_split = data_split
        self.train_ims_pth = loc
        # Captions---------------------------------------------
        def sort_elem(elem):
            elem_num = elem.split('.')
            return int(elem_num[0])

        self.captions_path = loc + data_split + '_sentence_list'
        self.captions = os.listdir(self.captions_path)
        self.captions.sort(key=sort_elem)

        self.length = len(self.captions)

        def sort_image(elem):
            elem_num = elem.split('.')
            return int(elem_num[0])

        self.images_path = loc + data_split + '_features'
        self.images = os.listdir(self.images_path)
        self.images.sort(key=sort_image)

        self.images_tag_path = loc + data_split + '_tags_vector'
        self.images_tag = os.listdir(self.images_tag_path)
        self.images_tag.sort(key=sort_image)

        self.boxes = np.load(loc + data_split + '_boxes.npy')

        print("Len in captions", self.length)
        self.public_data = True

        self.im_div = 5
        if self._get_data_split(data_split):
            if len(self.captions) > len(self.images):
                self.length = len(self.captions)
                self.im_div = len(self.captions) / len(self.images)
            else:
                self.length = len(self.images)
                self.im_div = len(self.images) / len(self.captions)
                self.public_data = False

    def _get_data_split(self, data_split):
        if data_split.startswith("test"):
            return True
        return False

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index / self.im_div)
        cap_id = index
        if not self.public_data:
            img_id = index
            cap_id = index / self.im_div
        image = torch.Tensor(parts.read_npy(os.path.join(self.images_path, self.images[img_id])))
        image_tag = torch.Tensor(parts.read_npy(os.path.join(self.images_tag_path, self.images_tag[img_id])))
        box = torch.Tensor(self.boxes[int(img_id)])

        target = parts.read_pickle(os.path.join(self.captions_path, self.captions[cap_id]))

        return image, image_tag, box, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[3]), reverse=True)
    images, images_tag, boxes, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    images_tag = torch.stack(images_tag, 0)
    boxes = torch.stack(boxes, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [cap.size(0) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), 768)
    for i, cap in enumerate(captions):
        for j, word_tensor in enumerate(cap):
            targets[i, j, :] = word_tensor[:]

    return images, images_tag, boxes, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
