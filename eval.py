### Evaluation K-Nearest Neighbours Approach
import tensorflow as tf
import load_data

class Evaluator():
	def __init__(self, opt, session):
		self.opt = opt
		self.dataset = load_data.Dataset(self.opt)

		#### Non-repeatable datasets required for testing
		train_dataset_full = self.dataset.create_dataset(set_name='train', repeat=False)
		val_dataset_full = self.dataset.create_dataset(set_name='val', repeat=False)

		





