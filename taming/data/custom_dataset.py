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

	def __init__(self, path_images: str, max_num_images=-1, sampling_rate=8_000, apply_transform: bool = False):
		print('init custom image-sketch dataset')
		self.path_images = path_images

		self.sampling_rate = sampling_rate

		self.check_dataset_folder()

		# get images
		print('scanning the images')
		self.raw_wave_paths = []
		for extension in ['wav', 'mp3', 'flac']:
			self.raw_wave_paths.extend(glob.glob(os.path.join(path_images, f'*.{extension}')))

		print('sorting the images and sketches')
		self.raw_wave_paths.sort()

		self.apply_transform = apply_transform

		if max_num_images > 0:
			print('limiting number of images')
			self.raw_wave_paths = self.raw_wave_paths[:max_num_images]
		print('done')

		self.size = len(self.raw_wave_paths)

		self.augmentations = SomeOf((1, None), [
			AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
			TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
			PitchShift(min_semitones=-12, max_semitones=12, p=0.5),
			HighPassFilter(min_cutoff_freq=600),
			OneOf([RoomSimulator(min_absorption_value=0.7, min_target_rt60=0.5),  # studio
			       RoomSimulator(min_absorption_value=0.15, min_target_rt60=0.9),  # office
			       RoomSimulator(min_absorption_value=0.05, min_target_rt60=1.5),  # Factory
			       RoomSimulator(min_absorption_value=1e-5, min_target_rt60=3),  # Extreme
			       TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=1.0)])
		])

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		path_wave = self.raw_wave_paths[idx]
		data, sr = librosa.load(path_wave, sr=self.sampling_rate)

		if self.apply_transform:
			data = self.transforms(data, sr)

		data = torch.tensor(data)  # .reshape(1, -1)

		if (diff := data.shape[0] % 16) > 0:
			data = F.pad(input=data, pad=(0, 16 - diff))

		return {"wav": data, 'sampling_rate': sr}

	def transforms(self, sample, sr):
		return self.augmentations(sample, sample_rate=sr)


class TrainLoader(AudioDataLoader):
	def __init__(self, path_wav: str, max_num_images=-1, sampling_rate=8_000, split=1, apply_transform=False):
		super().__init__(path_wav, max_num_images, sampling_rate, apply_transform)
		max_indx = int(split * self.__len__())
		self.raw_wave_paths = self.raw_wave_paths[:max_indx]
		self.size = max_indx


class TestLoader(AudioDataLoader):
	def __init__(self, path_wav: str, max_num_images=-1, sampling_rate=8_000, split=1, apply_transform=False):
		super().__init__(path_wav, max_num_images, sampling_rate, apply_transform)
		max_indx = int((1 - split) * self.__len__())
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
