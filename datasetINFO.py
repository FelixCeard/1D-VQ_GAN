from omegaconf import OmegaConf
from VQ_train_utils import instantiate_from_config
import numpy as np

if __name__ == '__main__':

	# load configs
	print('loading configs')
	configs = [OmegaConf.load('configs/JustDataloader.yaml')]
	config = OmegaConf.merge(*configs)

	# data
	print('loading data')
	data = instantiate_from_config(config.data)
	data.prepare_data()
	data.setup()

	lengths = []
	# for batch in data.train_dataloader()._get_iterator():
	# 	lengths.append(batch['spec'].shape[-1])

	# for batch in data.val_dataloader()._get_iterator():
	# 	lengths.append(batch['spec'].shape[-1])

	for batch in data.test_dataloader()._get_iterator():
		lengths.append(batch['spec'].shape[-1])


	lengths = np.array(lengths)
	print(np.mean(lengths), np.var(lengths))