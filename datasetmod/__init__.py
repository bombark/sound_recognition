#=================================  HEADER  ====================================

import os
import sys
import librosa
import numpy as np
import librosa.display
import pickle


labels = [
	"negative/checked"
	, "accelerating/1_New"
	, "accelerating/2_CKD_Long"
	, "accelerating/3_CKD_Short"
	, "accelerating/4_Old"
	, "braking/1_New"
	, "braking/2_CKD_Long"
	, "braking/3_CKD_Short"
	, "braking/4_Old"
]

tbl_id2label = {
	"negative": {
		"checked": 0
	}
	, "accelerating": {
		"1_New": 1
		, "2_CKD_Long": 2
		, "3_CKD_Short": 3
		, "4_Old": 4
	}
	, "braking": {
		"1_New": 5
		, "2_CKD_Long": 6
		, "3_CKD_Short": 7
		, "4_Old": 8
	}
}

freq_max = 6000


#-------------------------------------------------------------------------------



#===========================  PUBLIC FUNCTIONS  ================================

def id2label(id):
	return labels[id]

def label2id(label,sublabel):
	return tbl_id2label[label][sublabel]


def normalize(type, freq_res, time):
	print(
		"- Normalizing the dataset to (type: {}, freq_res: {}, time: {})".format
		(type, freq_res, time), file=sys.stderr
	)

	dataset_url = _get_dataset_path(type, freq_res, time)
	if os.path.exists( os.path.join(dataset_url,".ok") ):
		return;

	root_proj  = _find_project_root()
	root_dbraw = os.path.join( root_proj, "dataset" )
	root_mod   = os.path.join( root_proj, "datasetmod" )
	root_norm  = os.path.join( root_mod, "{}-{}-{}".format(type,freq_res,time) )
	os.makedirs( root_norm, exist_ok=True)

	for klass in ["accelerating","braking","negative"]:
		for subklass in os.listdir( os.path.join(root_dbraw,klass) ):
			print("\t", klass, subklass, file=sys.stderr)
			id = 0
			sum_duration = 0.0
			sum_descsize = 0.0
			klass_url = os.path.join(root_dbraw,klass,subklass)
			for file in os.listdir( klass_url ):
				id += 1
				url_sound = os.path.join(klass_url,file)
				image, duration, size = load_and_divide(url_sound, type, freq_res, time)
				serialized = pack_and_serialize(klass, subklass, image)
				dst_name = "{}/{}-{}-{:04}.pickle".format(root_norm,klass,subklass,id)
				with open(dst_name,"wb") as fd:
					fd.write( serialized )
				sum_duration += duration
				sum_descsize += size

			avg_duration = sum_duration/id
			avg_size = sum_descsize/id
			print("\t\tamount: {}, duration avg: {:.2f}s, size avg: {:.2f}".format
				(id,avg_duration,avg_size), file=sys.stderr
			)

	with open( os.path.join(root_norm, ".ok"), "w" ) as fd:
		fd.write("ok")


def load_data(type, freq_res, time):
	dataset_url = _get_dataset_path(type, freq_res, time)
	if not os.path.exists( os.path.join(dataset_url,".ok") ):
		normalize(type, freq_res, time)

	print("- Loading the dataset (type: {}, freq_res: {}, time: {})".format
		(type, freq_res, time), file=sys.stderr
	)
	_x_data = []
	y_data = []
	dataset_url = _get_dataset_path(type, freq_res, time)
	for file in os.listdir(dataset_url):
		if ( file == ".ok" ): continue;
		pickle_fd = open(os.path.join(dataset_url,file),"rb")
		pack = pickle.load(pickle_fd)
		klass_id = label2id( pack["class"], pack["subclass"] )
		for window in pack["data"]:
			_x_data.append( window )
			y_data.append(  klass_id )

	x_data = np.array( _x_data )
	#print( x_data.shape, len(y_data))
	return y_data, x_data


def load_realfile( url_sound, freq_res ):
	y, sr = librosa.load(url_sound)
	duration = librosa.get_duration(y=y, sr=sr)
	mel = librosa.feature.melspectrogram(
		y=y, sr=sr, n_mels=freq_res, fmax=freq_max
	)
	return mel, duration


#-------------------------------------------------------------------------------



#===========================  PRIVATE FUNCTIONS  ===============================

def _get_dataset_path(type,freq_res,time):
	return os.path.join( _find_project_root(), "datasetmod/{}-{}-{}".format
		(type,freq_res,time)
	)

def _find_project_root():
	cur = "./"
	found = False
	for i in range(5):
		if os.path.exists( os.path.join(cur,".root") ):
			found = True
			break
		cur = os.path.join(cur,"..")
	if not found:
		raise BaseException("file .root not found")
	return cur #os.path.join( cur, "dataset" )



def line_sum_of_image(image,line):
	max = 0.0
	shape = image.shape
	for i in range(shape[1]):
		sum += float(image[line,i])
	return sum/shape[1]



def load_and_divide(url_sound, type, freq_res=250, max_cols=125):
	# open the sound
	y, sr = librosa.load(url_sound)
	duration = librosa.get_duration(y=y, sr=sr)

	# build the descriptor
	if type == "mel":
		mel = librosa.feature.melspectrogram(
			y=y, sr=sr, n_mels=freq_res, fmax=freq_max
		)
	elif type == "stft":
		mel = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
	else:
		raise BaseException("type must be 'mel' or 'stft'")

	# divide the descritor in fixed windows
	cols = mel.shape[1]
	if cols < max_cols:
		norm = np.zeros((freq_res, max_cols))
		norm[0:freq_res,0:cols] = mel[0:freq_res,0:cols]
		return [ norm ], duration, cols  #  [ 250x125 ]

	elif cols == max_cols or cols < (max_cols+max_cols//2):
		norm = np.zeros((freq_res, max_cols))
		norm[0:freq_res,0:max_cols] = mel[0:freq_res,0:max_cols]
		return [ norm ], duration, cols  # [ 250x125 ]

	else:
		res = []
		for i in range( 0, cols-(max_cols//2), max_cols//2 ):
			#print(i,i+max_cols)
			if i+max_cols > cols:
				break
			norm = np.zeros((freq_res, max_cols))
			norm[0:freq_res,0:max_cols] = mel[0:freq_res,i:i+max_cols]
			res.append( norm )
		return res, duration, cols


def pack_and_serialize(klass,subklass,image):
	data = {}
	data["class"]    = klass
	data["subclass"] = subklass
	data["data"]     = image
	return pickle.dumps(data)

#-------------------------------------------------------------------------------
