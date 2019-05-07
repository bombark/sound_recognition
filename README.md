# sound_recognition


# Dependencies

pip3 install tensorflow keras numpy sklearn argparse librosa


# Install the dataset

mv tram_demo.tar.gz ./project/
cd ./project/
tar xvf tram_demo.tar.gz

# Execute

cd ./project/
./program ./test_files/tram-2018-11-30-15-30-17.wav [-c cnn|nn|knn|svm ]
