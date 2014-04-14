
Uni: nr2483

The bundle contains the following files
	
	data_extraction.py
	News_classifier.py
	news.txt
	README.txt

Instructions.

1) Install the required scikit libraries. Detailed instructions are available at:
	http://scikit-learn.org/stable/install.html

2) Save the news.txt file on disk

3) Run the data_extraction.py script and provide the required inputs such as file path of corpus when prompted by the system

4) Run the News_classifier.py script and provide the required inputs such as number of examples to be used for training when prompted by the system

The system makes use of Scikit (sklearn libraries)
Please find below a link which provides detailed instructions for installing scikit:
http://scikit-learn.org/stable/install.html
needs the following import statements to import the required libraries :
	import csv
	import numpy as np
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.metrics import confusion_matrix
	from sklearn.linear_model import SGDClassifier
	from sklearn.preprocessing import LabelBinarizer
	from sklearn.pipeline import  Pipeline
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import LinearSVC
	from sklearn.neighbors.classification import KNeighborsClassifier
	import time



news.txt (This is the corpus that is used for the project, this file can be stored at any desired path. When the data_extraction.py script is run, the system prompts the user to enter the path of the location where this file is stored.


Python Script Files:
data_extraction.py
News_Classifier.py


The path for the input .txt corpus file is taken from the user
The data extraction script extracts the relevant data from the corpus and creates a CSV corpus.csv to store the data in a format that can be used by the system


Methods:

1)__init__(self)
The init method is used to initialize various class variables.
It also reads the data from corpus.csv and shuffles the corpus to produce a random training and test data split based on the training sample size specified by the user

2)KnnClassifier(self)
implements the KnnClassifier and returns a list of classified output labels for the test data
3)SVM_LinearSVC(self)
implements the Linear SVC SVM Classifier and returns a list of classified output labels for the test data

4)SVM_SGDC_Classifier(self)
implements the Stochastic Gradient Descent Classifier and returns a list of classified output labels for the test data


Important Variables:

X_train = examples used for training (1 to train_ex from shuffled corpus)
Y1 	= labels corresponding to samples in X_train 
Y_train = labels corresponding to X_train + X_test
X_test 	= examples used for testing (train_ex to size)
y 	= Transformed output from label binarizer (lb)	 
