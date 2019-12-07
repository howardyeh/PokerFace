import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
# import sys
# print(sys.path)
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


IMG_SIZE = 224

class Preprocess:
	def __init__(self, _text_dir, _image_dir, batch_size=10):
		self.text_dir = _text_dir
		self.image_dir = _image_dir
		self.total_data_num = 0
		self.train_data_num = 0
		self.valid_data_num = 0
		self.train_data = []
		self.valid_data = []
		self.creat_all_data()
		self.batch_size = batch_size
		self.steps_per_epoch = self.train_data_num // batch_size
		

	# will be called once an epoch
	def shuffle_data(self):
		random.shuffle(self.train_data)


	def create_batch(self, step = 0,  dataset='train'):
		if step % (self.steps_per_epoch) == 0:
			self.shuffle_data()

		if dataset == 'train':

			step_in_epoch = step % (self.steps_per_epoch)
			batch_img = self.train_data[step_in_epoch * self.batch_size][0]
			batch_img = self.random_flip(batch_img)
			batch_img = cv2.resize(batch_img, (IMG_SIZE, IMG_SIZE))
			batch_img = self.get_random_crop(batch_img, int(IMG_SIZE * 0.8), int(IMG_SIZE * 0.8))
			batch_img = cv2.resize(batch_img, (IMG_SIZE, IMG_SIZE))
			batch_img = np.expand_dims(batch_img, axis=0)
			batch_predict = self.train_data[step_in_epoch * self.batch_size][1]
			batch_predict = np.expand_dims(batch_predict, axis=0)
			batch_label = self.train_data[step_in_epoch * self.batch_size][2]
			batch_label = np.expand_dims(batch_label, axis=0)

			for i in range(step_in_epoch*self.batch_size+1, step_in_epoch*self.batch_size + self.batch_size):
				temp_img = self.train_data[i][0]
				temp_img = self.random_flip(temp_img)
				temp_img = cv2.resize(temp_img, (IMG_SIZE, IMG_SIZE))
				temp_img = self.get_random_crop(temp_img, int(IMG_SIZE * 0.8), int(IMG_SIZE * 0.8))
				temp_img = cv2.resize(temp_img, (IMG_SIZE, IMG_SIZE))
				temp_img = np.expand_dims(temp_img, axis=0)
				batch_img = np.append(batch_img, temp_img, axis = 0)
				temp_predict = self.train_data[i][1]
				temp_predict = np.expand_dims(temp_predict, axis=0)
				batch_predict = np.append(batch_predict, temp_predict, axis = 0)
				temp_label = self.train_data[i][2]
				temp_label = np.expand_dims(temp_label, axis=0)
				batch_label = np.append(batch_label, temp_label, axis = 0)

			print("batch_img shape = ",batch_img.shape)
			print("batch_predict shape = ", batch_predict.shape)
			print("batch_label shape = ", batch_label.shape)
			return batch_img, batch_predict, batch_label

		if dataset == 'validation':
			batch_img = self.valid_data[step * self.batch_size][0]
			batch_img = cv2.resize(batch_img, (IMG_SIZE, IMG_SIZE))
			batch_img = np.expand_dims(batch_img, axis=0)
			batch_predict = self.valid_data[step * self.batch_size][1]
			batch_predict = np.expand_dims(batch_predict, axis=0)
			batch_label = self.valid_data[step * self.batch_size][2]
			batch_label = np.expand_dims(batch_label, axis=0)

			for i in range(step*self.batch_size+1, step*self.batch_size + self.batch_size):
				temp_img = self.valid_data[i][0]
				temp_img = cv2.resize(temp_img, (IMG_SIZE, IMG_SIZE))
				temp_img = np.expand_dims(temp_img, axis=0)
				batch_img = np.append(batch_img, temp_img, axis = 0)
				temp_predict = self.valid_data[i][1]
				temp_predict = np.expand_dims(temp_predict, axis=0)
				batch_predict = np.append(batch_predict, temp_predict, axis = 0)
				temp_label = self.valid_data[i][2]
				temp_label = np.expand_dims(temp_label, axis=0)
				batch_label = np.append(batch_label, temp_label, axis = 0)
			
			return batch_img, batch_predict, batch_label

	def creat_all_data(self):
		f = open(self.text_dir, "r")
		all_data = []
		for line in f:
			self.total_data_num += 1
			line = line.strip('\n')
			a = line.split(' ')
			img_array_bgr = cv2.imread(os.path.join(self.image_dir, a[0]))
			img_array_rgb = img_array_bgr[:, :, ::-1]
			all_data.append([img_array_rgb, np.array([a[1]]), np.array([a[2]])])
		f.close()

		random.shuffle(all_data)
		for i in range(int(self.total_data_num*0.7)):
			self.train_data_num += 1
			self.train_data.append(all_data[i])
		valid_data = []
		for i in range(int(self.total_data_num*0.7), self.total_data_num):
			self.valid_data_num += 1
			self.valid_data.append(all_data[i])
		return
		

	def get_random_crop(self, image, crop_height, crop_width):
		max_x = image.shape[1] - crop_width
		max_y = image.shape[0] - crop_height

		x = np.random.randint(0, max_x)
		y = np.random.randint(0, max_y)

		crop = image[y: y + crop_height, x: x + crop_width]

		return crop

	def random_flip(self, image):
		flip = random.randint(0, 2)
		if flip == 1:
			return image
		else:
			return cv2.flip(image, 1)

	
