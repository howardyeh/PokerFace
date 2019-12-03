import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle 


IMG_SIZE = 128

class Preprocess:
	def __init__(self, _text_dir, _image_dir, batch_size=10):
		"""
		assign data file direction
		"""
		self.text_dir = _text_dir
		self.image_dir = _image_dir
		self.X = []
		self.predict = []
		self.label = []
		self.all_data = self.creat_all_data()
		self.list_img = sorted(os.listdir(self.image_dir))
		self.batch_size = batch_size
		self.size = len(self.list_img)
		self.num_batch = self.size // batch_size
		

	# will be called once an epoch
	def shuffle_data(self):
		np.random.shuffle(self.all_data)
    	for img, value in all_data:
        	self.X.append(img)
        	self.predict.append(float(value[0]))
        	self.label.append(float(value[1]))

	def create_batch(self, dataset='train', step = 0):
		if step % (self.num_batch * 0.8) == 0:
			self.X = []
			self.predict = []
			self.label = []
			self.shuffle_data()

		batch_idx = step % (self.num_batch * 0.8)
		training_img = self.X[: self.size * 0.8]
		training_predict = self.predict[: self.size * 0.8]
		testing_label = self.label[: self.size * 0.8]
		testing_img = self.X[self.size * 0.8:]
		testing_predict = self.predict[self.size * 0.8:]
		testing_label = self.label[self.size * 0.8:]

		if dataset == 'train':
			if batch_idx == self.num_batch * 0.8:
				batch_img = training_img[batch_idx*self.batch_size:]
				batch_predict = training_predict[batch_idx*self.batch_size:]
				batch_label = training_label[batch_idx*self.batch_size:]

			else:
				batch_img = training_img[batch_idx*self.batch_size: (batch_idx + 1)*self.batch_size]
				batch_predict = training_predict[batch_idx*self.batch_size: (batch_idx + 1)*self.batch_size]
				batch_label = training_label[batch_idx*self.batch_size: (batch_idx + 1)*self.batch_size]

			batch_img = np.array(batch_img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
			batch_predict = np.array(batch_predict)
			batch_label = np.array(batch_label)
			
			return batch_img, batch_predict, batch_label

		if dataset == 'test':
			# if step > self.num_batch * 0.2:
			# 	return None, None, None

			batch_img = testing_img[step * self.batch_size: (step + 1) * self.batch_size]
			batch_predict = testing_predict[step * self.batch_size: (step + 1) * self.batch_size]
			batch_label = testing_label[step * self.batch_size: (step + 1) * self.batch_size]

			batch_img = np.array(batch_img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
			batch_predict = np.array(batch_predict)
			batch_label = np.array(batch_label)
			
			return batch_img, batch_predict, batch_label

	def creat_all_data(self):
		f = open(text_dir, "r")
		image_predict_label = {}
		for line in f:
			line = line.strip('\n')
			a = line.split(' ')
			image_predict_label[a[0]] = [a[1], a[2]]
		f.close()

		all_data = []
    	for img in os.listdir(self.image_dir):
        	if os.path.splitext(img)[-1] == ".png":
            	img_array_bgr = cv2.imread(os.path.join(image_dir, img))
            	img_array_rgb = img_array_bgr[:, :, ::-1]
            	img_resize = cv2.resize(img_array_rgb, (IMG_SIZE, IMG_SIZE))
            	all_data.append([img_resize, image_predict_label[img]])

        for img, value in all_data:
        	self.X.append(img)
        	self.predict.append(float(value[0]))
        	self.label.append(float(value[1]))

         return all_data

	
