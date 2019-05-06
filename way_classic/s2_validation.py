#!/usr/bin/python3

#=================================  HEADER  ====================================

import sys
import datasetmod as dataset
import pickle
import argparse
import s1_training

from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.metrics         import accuracy_score

#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def exec(clf_name, desc, freq_res, time):
	# load the training data
	y_data, _x_data = dataset.load_data(desc, freq_res, time)
	x_data = _x_data.reshape( len(_x_data), freq_res*time )
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

	# load or training the classifer
	clf = s1_training.exec(clf_name, desc, freq_res, time)

	# Predict and show the results
	print("- Validating the classifier (type: {}, freq_res: {}, time: {})".format(clf_name,freq_res,time), file=sys.stderr )
	print('\t test size:',len(x_test), file=sys.stderr)
	y_pred = clf.predict(x_test)
	print(confusion_matrix(y_test, y_pred), file=sys.stderr )
	report = classification_report(y_test, y_pred)
	print(report, file=sys.stderr)

	# return accuracy
	return accuracy_score(y_test, y_pred)


#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-c','--cfs', help='knn, svm or nn', default='svm')
	parser.add_argument('-d','--desc', help='mel or stft', default='mel')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=155)
	args = parser.parse_args()
	acc = exec(args.cfs,args.desc,int(args.freq_res),int(args.time))
	print( acc )

#-------------------------------------------------------------------------------
