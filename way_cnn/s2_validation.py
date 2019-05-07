#!/usr/bin/python3

#=================================  HEADER  ====================================

import sys
import datasetmod as dataset
import argparse
import tensorflow as tf
import numpy      as np
#import way_cnn.s1_training

from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score


#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def exec(desc, freq_res, time):
	# normalize the dataset, case it dont exist
	model = s1_training.exec(desc,freq_res,time)

	# load test data
	y_data, x_data = dataset.load_data(desc,freq_res,time)
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
	x_test = x_test.reshape(len(x_test), freq_res, time, 1)

	# evaluate
	y_pred = []
	y_pred_cnn = model.predict(x_test)
	for i in range(len(x_test)):
		y_pred.append( np.argmax(y_pred_cnn[i]) )
		#print(y_test[i], , max(y_pred[i]) )

	print( confusion_matrix(y_test, y_pred), file=sys.stderr )
	print( classification_report(y_test, y_pred), file=sys.stderr )

	# return accuracy
	return accuracy_score(y_test, y_pred)

	#confusion_matrix = tf.confusion_matrix(["a","b","c","d","e","f","g","h"], predictions)
	#print(confusion_matrix)

#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	import s1_training
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-d','--desc', help='mel or stft', default='mel')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=150)
	args = parser.parse_args()
	acc = exec(args.desc, int(args.freq_res), int(args.time))
	print(acc)

else:
	from way_cnn import s1_training


#-------------------------------------------------------------------------------
