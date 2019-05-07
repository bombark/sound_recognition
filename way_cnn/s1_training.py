#!/usr/bin/python3

#=================================  HEADER  ====================================

import sys
import os
import datasetmod as dataset
import argparse

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection    import train_test_split

#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def make_model(freq_res, time):
	# Simple model
	model = Sequential([
		Flatten(input_shape=(freq_res, time, 1)),
		#Dense(16, activation=tf.nn.relu), #512
		BatchNormalization(),
		Dense(9, activation=tf.nn.softmax)
	])

	# piczak model
	# model = Sequential([
	# 	Conv2D(80, kernel_size=(12, 6), activation='relu', input_shape=(freq_res,time,1)),
	# 	MaxPooling2D(pool_size=(6, 3)),
	# 	Flatten(),
	# 	#MaxPooling2D(pool_size=(1, 3)),
	# 	BatchNormalization(),
	# 	Dense(9, activation=tf.nn.softmax)
	# ])
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	return model



def exec(desc, freq_res, time):
	# make the model and if exists already, load the weights
	model_url = "./checkpoints/{}-{}-{}".format(desc,freq_res,time)
	model = make_model(freq_res,time)
	if os.path.exists(model_url):
		model.load_weights('./checkpoints/{}-{}-{}'.format(desc, freq_res, time))
		return model

	# load the training data
	y_data, x_data = dataset.load_data(desc,freq_res,time)
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

	# training the model
	print("- Training a CNN", file=sys.stderr)
	x_train = x_train.reshape(len(x_train),freq_res,time,1)

	print(x_train.shape, file=sys.stderr)
	model.fit(
		x_train, y_train
		, epochs = 10
		, verbose=1
	)

	# save the model
	os.makedirs( "checkpoints", exist_ok=True)
	model.save_weights('./checkpoints/{}-{}-{}'.format(desc, freq_res, time))

	return model

#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-d','--desc', help='mel or stft', default='mel')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=150)
	args = parser.parse_args()
	exec(args.desc, int(args.freq_res), int(args.time))

#-------------------------------------------------------------------------------
