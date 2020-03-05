"""
Run command:
python3 log_reg.py fft_extracted 
"""

import sklearn 
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np


def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that math genre-dir
		file_list = glob.glob(genre_dir)
		#print(file_list)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)
	
	#print(X[0])
	#print(len(X))
	#print(len(y))

	return np.array(X), np.array(y)



def log_reg_func(X_train, y_train, X_test, y_test, genre_list):

	print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	logistic_classifier = linear_model.LogisticRegression()
	logistic_classifier.fit(X_train, y_train)
	logistic_predictions = logistic_classifier.predict(X_test)
	logistic_accuracy = accuracy_score(y_test, logistic_predictions)
	logistic_cm = confusion_matrix(y_test, logistic_predictions)
	print("logistic accuracy = " + str(logistic_accuracy))
	print("logistic_cm :")
	print(logistic_cm)
	joblib.dump(logistic_classifier, 'model.pkl')
	print("Model Saved\n")
	
	
	plot_confusion_matrix(logistic_cm, "Confusion matrix", genre_list)
	# plot_confusion_matrix(knn_cm, "Confusion matrix for FFT classification", genre_list)


def plot_confusion_matrix(cm, title, genre_list, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('fft_confusion_matrix.png')
    print("Confusion Matrix saved")
    #plt.show()


def main():
	# first command line argument is the base folder that consists of the fft files for each genre
	base_dir_fft  = sys.argv[1]
	
	"""list of genres (these must be folder names consisting .wav of respective genre in the base_dir)
	"""
	genre_list = os.listdir('./gtzan')
	


	# use FFT
	X, y = read_fft(genre_list, base_dir_fft)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

	# print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	
	print('\nUsing FFT')
	log_reg_func(X_train, y_train, X_test, y_test, genre_list)




if __name__ == "__main__":
	main()
