# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from enum import Enum, unique

from .binvox_rw import read_as_3d_array
import logging
from .logging import debug, info, warn, error

@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2

# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        '''if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]'''
        selected_rendering_image_paths = [
            rendering_image_paths[i]
            for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
        ]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, args, rotation=False, half=False, pix3d=False):
        self.dataset_taxonomy = None
        self.pix3d = pix3d
        if rotation:
            if half:
                self.rendering_image_path_template = args.testset_path_half
            else:
                self.rendering_image_path_template = args.testset_path_all
        else:
            if pix3d:
                self.rendering_image_path_template = args.trainset_pix3d_path
            else:
                self.rendering_image_path_template = args.trainset_path
        self.volume_path_template = args.voxel_path
        if torch.distributed.get_rank() == 0:
            logging.info(self.rendering_image_path_template)

        # Load all taxonomies of the dataset
        with open(args.taxonomy_path, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None, n_samples=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            if self.pix3d:
                if taxonomy_folder_name != '03001627':
                    continue
            info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                         (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples, n_samples))

        random.shuffle(files)

        info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering,transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples, n_samples=None):
        files_of_taxonomy = []
        random.shuffle(samples)

        for sample_idx, sample_name in enumerate(samples):
            if n_samples is not None and sample_idx >= n_samples:
                break
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(volume_file_path):
                warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


class Pix3dDataset(torch.utils.data.dataset.Dataset):
    """Pix3D class used for PyTorch DataLoader"""
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, bounding_box = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images, bounding_box)

        return taxonomy_name, sample_name, rendering_images, volume

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_path = self.file_list[idx]['rendering_image']
        bounding_box = self.file_list[idx]['bounding_box']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        rendering_image = cv2.imread(rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if len(rendering_image.shape) < 3:
            warn('It seems the image file %s is grayscale.' % (rendering_image_path))
            rendering_image = np.stack((rendering_image, ) * 3, -1)

        # Get data of volume
        with open(volume_path, 'rb') as f:
            volume = read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray([rendering_image]), volume, bounding_box

class Pix3dDataLoader:
    def __init__(self, agrs):
        self.dataset_taxonomy = None
        self.annotations = dict()
        self.volume_path_template = '/home/ubuntu/zhouhang/pix3d/model/%s/%s/%s.binvox'
        self.rendering_image_path_template = '/home/ubuntu/zhouhang/pix3d/img/%s/%s.%s'

        # Load all taxonomies of the dataset
        with open('pix2vox/datasets/Pix3D.json', encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

        # Load all annotations of the dataset
        _annotations = None
        with open('/home/ubuntu/zhouhang/pix3d/pix3d.json', encoding='utf-8') as file:
            _annotations = json.loads(file.read())

        for anno in _annotations:
            filename, _ = os.path.splitext(anno['img'])
            anno_key = filename[4:]
            self.annotations[anno_key] = anno

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_name = taxonomy['taxonomy_name']
            info('Collecting files of Taxonomy[Name=%s]' % (taxonomy_name))

            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_name, samples))

        info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return Pix3dDataset(files, transforms)

    def get_files_of_taxonomy(self, taxonomy_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get image annotations
            anno_key = '%s/%s' % (taxonomy_name, sample_name)
            annotations = self.annotations[anno_key]

            # Get file list of rendering images
            _, img_file_suffix = os.path.splitext(annotations['img'])
            rendering_image_file_path = self.rendering_image_path_template % (taxonomy_name, sample_name,
                                                                              img_file_suffix[1:])

            # Get the bounding box of the image
            img_width, img_height = annotations['img_size']
            bbox = [
                annotations['bbox'][0] / img_width,
                annotations['bbox'][1] / img_height,
                annotations['bbox'][2] / img_width,
                annotations['bbox'][3] / img_height
            ]  # yapf: disable
            model_name_parts = annotations['voxel'].split('/')
            model_name = model_name_parts[2]
            volume_file_name = model_name_parts[3][:-4].replace('voxel', 'model')

            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_name, model_name, volume_file_name)
            if not os.path.exists(volume_file_path):
                print(volume_file_path)
                warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_name,
                'sample_name': sample_name,
                'rendering_image': rendering_image_file_path,
                'bounding_box': bbox,
                'volume': volume_file_path,
            })

        return files_of_taxonomy

class ABODataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return taxonomy_name, sample_name, rendering_images, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']

        # Get data of rendering images
        '''if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]
        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]'''
        selected_rendering_image_paths = [
            rendering_image_paths[i]
            for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
        ]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume

class ABODataLoader:
    def __init__(self, args):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = args.abo_path
        self.volume_path_template = args.abo_voxel_path

        # Load all taxonomies of the dataset
        with open(args.abo_taxonomy_path, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None, n_samples=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']
            info('Collecting files of Taxonomy[ID=%s, Name=%s]' %
                         (taxonomy['taxonomy_id'], taxonomy['taxonomy_name']))
            samples = []
            if dataset_type == DatasetType.TRAIN:
                samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['test']

            files.extend(self.get_files_of_taxonomy(taxonomy_folder_name, samples, n_samples))

        random.shuffle(files)

        info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return ShapeNetDataset(dataset_type, files, n_views_rendering,transforms)

    def get_files_of_taxonomy(self, taxonomy_folder_name, samples, n_samples=None):
        files_of_taxonomy = []
        random.shuffle(samples)

        for sample_idx, sample_name in enumerate(samples):
            if n_samples is not None and sample_idx >= n_samples:
                break
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name, sample_name)
            if not os.path.exists(volume_file_path):
                warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_images_file_path,
                'volume': volume_file_path,
            })

        return files_of_taxonomy


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #

