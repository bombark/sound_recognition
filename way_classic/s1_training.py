#!/usr/bin/python3


#=================================  HEADER  ====================================

# article
# http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf
# https://www.youtube.com/watch?v=MhOdbtPhbLU

import os
import datasetmod as dataset
import pickle
import argparse

from sklearn                 import svm
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def exec(clf_name, descritor, freq_res, time):
	# Load the classifier, case it exists
	os.makedirs( "checkpoints", exist_ok=True)
	clf_url = "checkpoints/{}-{}-{}.pickle".format(clf_name,freq_res,time)
	if os.path.exists(clf_url):
		with open(clf_url,"rb") as fd:
			clf = pickle.loads(fd.read())
		return clf

	# Training a new classifier
	# load the training data
	y_data, _x_data = dataset.load_data(descritor,freq_res,time)
	x_data = _x_data.reshape( len(_x_data), freq_res*time )
	#print(x_data.shape)
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

	# make the model and training
	print("- Training a {} ".format(clf_name) )
	print("\t","train: ",len(x_train), "test:", len(x_test))
	if clf_name == "svm":
		clf = svm.SVC(kernel='linear')
	elif clf_name == "knn":
		clf = KNeighborsClassifier()
	elif clf_name == "nn":
		clf = MLPClassifier()
	else:
		raise BaseException("clf_name must be 'svm', 'knn' or 'nn'")
	clf.fit(x_train, y_train)

	# save the model
	serialized = pickle.dumps(clf)
	with open( clf_url , "wb" ) as fd:
		fd.write(serialized)

	return clf

#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-c','--cfs', help='knn, svm or nn', default='svm')
	parser.add_argument('-d','--desc', help='mel or stft', default='mel')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=150)
	args = parser.parse_args()
	exec(args.cfs, args.desc, int(args.freq_res), int(args.time))

#-------------------------------------------------------------------------------
