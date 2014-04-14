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

class News_Classifier:
    X_train=[]
    X_test=[]
    Y_train=[]
    y=[]
    Y1=[]
    size=0
    train_ex=0
    
    def __init__(self):
        #List to store input text
        data_input=[]
        #List to store output labels
        data_output=[]
        
        train_text=[]
        
        self.train_ex=int(raw_input('Please enter the number of examples that should be used to train the model\n'))

        with open('corpus.csv','r') as f:
            train_csv=csv.reader(f)
            
            self.size=-1
            for row in train_csv:
                if self.size==-1:
                    self.size=0
                else:
                    self.size=self.size+1
                    data_input.append(row[1])
                    data_output.append(row[6])
        print 'There are ',self.size,' examples in the corpus\n'    
          
        #Generate a  permutation to re-shuffle the corpus so that training and testing data can be split randomly          
        perm=np.random.permutation(self.size)
        
        #Shuffle the entire corpus            
        for p in perm:
            train_text.append(data_input[p])
            self.Y_train.append(data_output[p])
        
       
        
        
        self.X_train = np.array(train_text[:self.train_ex])
        self.X_test  = np.array(train_text[self.train_ex:self.size])
        
        self.lb=LabelBinarizer()
        self.Y1=self.Y_train[:self.train_ex]
        self.y = self.lb.fit_transform(self.Y1)
            
    def KnnClassifier(self):        
        KNN_classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', KNeighborsClassifier(neighbors=5))
                ])
         
        KNN_classifier.fit(self.X_train,self.y)
         
        predicted = KNN_classifier.predict(self.X_test)
        y_pred = self.lb.inverse_transform(predicted)
         
        i=self.train_ex
        correct=0
        for label in y_pred:
            if label==self.Y_train[i]:
                correct=correct+1
            i = i + 1
            
        print 'Number of Examples used for Training',self.train_ex
        print 'Number of Correctly classified',correct
        print 'Total number of samples classified in Test data',self.size-self.train_ex
        print 'The resulting accuracy using KNN is ',(float(correct)*100/float(self.size-self.train_ex)),'%\n'
        cm=confusion_matrix(self.Y_train[self.train_ex:self.size],y_pred)
        print 'The confusion matrix is',cm
        return y_pred
     
    def SVM_LinearSVC(self):        
        SVM_Classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(LinearSVC()))
                ])
         
        SVM_Classifier.fit(self.X_train,self.y)
         
        predicted = SVM_Classifier.predict(self.X_test)
        y_pred = self.lb.inverse_transform(predicted)
         
        i=self.train_ex
        correct=0
        for label in y_pred:
            if label==self.Y_train[i]:
                correct=correct+1
            i = i + 1
            
        print 'Number of Examples used for Training',self.train_ex
        print 'Number of Correctly classified',correct
        print 'Total number of samples classified in Test data',self.size-self.train_ex
        print 'The resulting accuracy using Linear SVC is ',(float(correct)*100/float(self.size-self.train_ex)),'%\n'

        cm=confusion_matrix(self.Y_train[self.train_ex:self.size],y_pred)
        print 'The confusion matrix is',cm
     
        return y_pred
    
    def SVM_SGDC_Classifier(self):        
        SGDC_Classifier = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', OneVsRestClassifier(SGDClassifier()))
                ])
         
        SGDC_Classifier.fit(self.X_train,self.y)
         
        predicted = SGDC_Classifier.predict(self.X_test)
        y_pred = self.lb.inverse_transform(predicted)
         
        i=self.train_ex
        correct=0
        for label in y_pred:
            if label==self.Y_train[i]:
                correct=correct+1
            i = i + 1
            
        print 'Number of Examples used for Training',self.train_ex
        print 'Number of Correctly classified',correct
        print 'Total number of samples classified in Test data',self.size-self.train_ex
        print 'The resulting accuracy using Stochastic Gradient Descent is ',(float(correct)*100/float(self.size-self.train_ex)),'%\n'
 
        cm=confusion_matrix(self.Y_train[self.train_ex:self.size],y_pred)
        print 'The confusion matrix is\n',cm
     
        return y_pred

start=time.time()
print 'Initializing....'
n=News_Classifier()
start=time.time()

print '\nRunning K Nearest Neighbors Classification'
n.KnnClassifier()
time2=time.time()
knn_time=time.time()-start

print '\nRunning SVM Classification'
n.SVM_LinearSVC()
time3=time.time()
svm_time=time.time()-start-knn_time

print '\nRunning Stochastic Gradient Descent Classification'
n.SVM_SGDC_Classifier()
time4=time.time()
sgdc_time=time.time()-start-svm_time

print '\nThe running time was ',time.time()-start, ' seconds'