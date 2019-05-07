#!/usr/bin/python3


#=================================  HEADER  ====================================

import sys
import argparse
import datasetmod  as dataset
import numpy       as np
import cv2
import math

#-------------------------------------------------------------------------------



#===============================  FUNCTIONS  ===================================

def sec2min(total_sec):
	sec = int(total_sec) % 60
	min = total_sec // 60
	return "{:02d}:{:02d}".format( int(min), int(sec) )


def show_line(time, id):
	print( sec2min(time), end='' )
	if id == 1:
		print(", 1",end='')
	else:
		print(", 0",end='')
	for i in range(2,len(dataset.labels)):
		if i == id:
			print(", 1",end='')
		else:
			print(", 0",end='')
	print("")


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

		# start the recognition
		last_id = 0
		last_count = 0
		last_time = 0


		sound_rgb = cv2.convertScaleAbs(sound, alpha=255, beta=30)
		sound_rgb = cv2.cvtColor(sound_rgb, cv2.COLOR_GRAY2RGB)

		res = np.zeros( (rows*2,cols,3), dtype=np.uint8 )
		res[0:rows,0:cols] = sound_rgb

		for i in range( 0, cols-time, 11 ):
			window = sound[0:rows, i:i+time]
			time_cur = i / sec

			# reshape the input data
			if classifier_name == "cnn":
				x_data = window.reshape(1,rows,time,1)
			else:
				x_data = window.reshape(1,rows*time)

			# predict
			_y =  classifier.predict( x_data )
			if classifier_name == "cnn":
				y = np.argmax(_y[0])
			else:
				y = _y[0]

			if y > 0:
				colors = [
					(0,0,0)
					, (0,64,0), (0,128,0), (0,192,0), (0,255,0)
					, (0,0,64), (0,0,128), (0,0,192), (0,0,255)
				]
				cv2.rectangle(res,(i,rows),(i+11,rows+(rows//2)), colors[y], -1)


		lines = math.ceil(cols//800)
		broked = np.zeros( (rows*2*lines,800,3), dtype=np.uint8 )
		for line in range(0,cols//800):
			ini_x = line*800
			ini_y = line*rows*2
			end_y = ini_y+rows*2
			broked[ini_y:end_y, 0:800] = res[:,ini_x:ini_x+800]

		cv2.imshow("w",broked)
		cv2.waitKey(0)


#-------------------------------------------------------------------------------



#==================================  MAIN  =====================================

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='sound recognition')
	parser.add_argument('wav', help='wav file')
	parser.add_argument('-c','--clf', help='cnn, svm, nn or knn', default='cnn')
	parser.add_argument('-f','--freq_res', help='frequency resolution', default=25)
	parser.add_argument('-t','--time', help='window size', default=150)
	args = parser.parse_args()
	exec(args.wav, args.clf, int(args.freq_res),int(args.time))

#-------------------------------------------------------------------------------
