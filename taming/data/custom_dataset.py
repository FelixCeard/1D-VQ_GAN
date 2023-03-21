#!/usr/bin/env python
"""
Custom dataset loader
"""
import glob
import os
import random

import librosa
import torch
import torch.nn.functional as F
from audiomentations import AddGaussianNoise, HighPassFilter, OneOf, PitchShift, RoomSimulator, SomeOf, TanhDistortion, \
	TimeStretch
from torch.utils.data import Dataset
import torchaudio.functional as Faudio
from torchaudio.transforms import FrequencyMasking, TimeStretch, PitchShift
import pandas as pd


class PathException(Exception):
	def __init__(self, string):
		super(PathException, self).__init__(string)


class AudioDataLoader(Dataset):
	"""
	Returns the pure audio dataset of a folder
	Goal is to be able to be compressed using a VQ-GAN to then be classified using a transformer
	"""

	def check_dataset_folder(self):
		if not os.path.isdir(self.path_images):
			raise PathException(f"The given path is not a valid: {self.path_images}")

	def __init__(self, path_images: str, max_num_images=-1, sampling_rate=8_000, apply_transform: bool = False, split:str = 'train', tsv_path='./../../dataset/SDR_metadata.tsv', speaker:str='all'):
		# print('init custom image-sketch dataset')
		self.path_images = path_images

		self.sampling_rate = sampling_rate
		self.max_num_images = max_num_images
		self.check_dataset_folder()

		# filtering the paths
		df = pd.read_csv(tsv_path, sep='	')
		df = df[df['split'] == split.upper()] # filter for the specific thing
		if speaker != 'all':
			df = df[df['speaker'] == speaker]
		paths = df['file'].tolist()
		paths = [os.path.join(path_images, p) for p in paths]

		# get images
		# print('scanning the images')
		self.raw_wave_paths = [p for p in paths if os.path.isfile(p)]

		# print('sorting the images and sketches')
		self.raw_wave_paths.sort()

		self.apply_transform = apply_transform

		if max_num_images > 0:
			print('limiting number of images')
			self.raw_wave_paths = self.raw_wave_paths[:max_num_images]

		self.size = len(self.raw_wave_paths)
		print('Found', self.size, 'audio samples for the split:', split)

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		path_wave = self.raw_wave_paths[idx]
		data, sr = librosa.load(path_wave, sr=self.sampling_rate)

		# transform
		# if random.random() > 0.6: # in 40% of the case

		data = torch.tensor(data)  # .reshape(1, -1)

		if (diff := data.shape[0] % 16) > 0:
			data = F.pad(input=data, pad=(0, 16 - diff))

		data = data[:12_000] # limit input to 1,5 seconds (rest is probably empty)

		label = int(path_wave.split('/')[-1][0])

		return {"wav": data, 'sampling_rate': sr, 'label':label}


class TrainLoader(AudioDataLoader):
	def __init__(self, path_wav: str, max_num_images=-1, sampling_rate=8_000, split=1, apply_transform=False):
		super().__init__(path_wav, max_num_images, sampling_rate, apply_transform, split='train')
		max_indx = int(split * self.__len__())

		if max_num_images <= 0:
			self.raw_wave_paths = self.raw_wave_paths[:max_indx]
			self.size = max_indx

class DatasetLoader(AudioDataLoader):
	def __init__(self, path_wav: str, max_num_images=-1, sampling_rate=8_000, split='train', tsv_path:str='.', speaker:str = 'all'):
		super().__init__(path_wav, max_num_images, sampling_rate, apply_transform=False, split=split, tsv_path=tsv_path, speaker=speaker)


class TestLoader(AudioDataLoader):
	def __init__(self, path_wav: str, max_num_images=-1, sampling_rate=8_000, split=1, apply_transform=False):
		super().__init__(path_wav, max_num_images, sampling_rate, apply_transform, split='TEST')
		max_indx = int((1 - split) * self.__len__())
		if max_num_images <= 0:
			self.raw_wave_paths = self.raw_wave_paths[max_indx:]
			self.size = max_indx


class RandomChoice(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.t = random.choice(self.transforms)

	def __call__(self, img):
		return self.t(img)


class RandomChoiceBatch(torch.nn.Module):
	def __init__(self, transforms):
		super().__init__()
		self.transforms = transforms

	def __call__(self, imgs):
		t = random.choice(self.transforms)
		return [t(img) for img in imgs]
