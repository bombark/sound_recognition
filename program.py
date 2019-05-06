#!/usr/bin/python3


#=================================  HEADER  ====================================

import sys
import argparse
import datasetmod  as dataset
import numpy       as np
import cv2

#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def sec2min(total_sec):
	sec = int(total_sec) % 60
	min = total_sec // 60
	return "{:02d}:{:02d}".format( int(min), int(sec) )


def exec(sound_url, classifier_name, freq_res, time):
		# train or restore the model and weights
		if classifier_name == "cnn":
			from   way_cnn     import s1_training   as training
			classifier = training.exec("mel",freq_res,time)
		else:
			from   way_classic import s1_training   as training
			classifier = training.exec(classifier_name, "mel",freq_res,time)

		# open the sound and convert to time-frequency domain
		sound, duration = dataset.load_realfile(sound_url, freq_res)
		rows = sound.shape[0]
		cols = sound.shape[1]
		sec  = cols / duration;

		last_id = 0
		last_count = 0
		for i in range( 0, cols-time, 11 ):
			window = sound[0:rows, i:i+time]
			time_cur = i / sec

			if classifier_name == "cnn":
				x_data = window.reshape(1,rows,time)
			else:
				x_data = window.reshape(1,rows*time)

			_y =  classifier.predict( x_data )
			if classifier_name == "cnn":
				y = np.argmax(_y[0])
			else:
				y = _y[0]

			if y != last_id:
				if last_id > 0 and last_count > 1:
					print( "{} {} {}".format(sec2min(time_cur), last_id, last_count) )
				last_id = y
				last_count = 1
			else:
				last_count += 1

			# if y > 0:
			# 	print(sec2min(time_cur),y)
			#	cv2.imshow("w",window)
			#	cv2.waitKey(1000)


		if last_id > 0:
			print( "{} {} {}".format(sec2min(time_cur), last_id, last_count) )


#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='sound recognition')
	parser.add_argument('wav', help='wav file')
	parser.add_argument('-c','--clf', help='cnn, svm, nn or knn', default='cnn')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=155)
	args = parser.parse_args()
	exec(args.wav, args.clf, int(args.freq_res),int(args.time))

#-------------------------------------------------------------------------------











#==================================  MAIN  =====================================

#
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('wav', type=str, help='an integer for the accumulator')
#
# parser.add_argument('-c', '--clf', help='knn, nn, svm, cnn', default="cnn")
#
# args = parser.parse_args()
#
# acc = []




#-------------------------------------------------------------------------------
