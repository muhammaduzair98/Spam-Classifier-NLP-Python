#Importing Dataset

import pandas as pd

#Reading Dataset and Adding Column By Seperating into 2 Cols with new labels as per Tab Indentation in file '\t'

messages = pd.read_csv('/Users/eapple/Desktop/Roman_Urdu_DataSet.csv',
                       names=["Message","Label", "Nan"])

selection = messages.iloc[:2,:2]
print(selection)


#Import Lib & Creating Objects
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

corpus = []

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#Cleaning Dataset

for i in range (0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) 
              for word in review 
              if not word in stopwords.words('german')]
    review = ' '.join(review)
    corpus.append(review)    
    

#Creating Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

#Labeling The LABEL Cols with Binary (Assigning Dummy Values For Better Understanding of Machine)

y=pd.get_dummies(messages['label'])
print(messages['label'])

#Consider only 1 Col
y=y.iloc[:,1].values

#Printing 1 Cols

print (y)



#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)

print ('X_Train \n \n', X_train, '\n')
print ('X_Test \n \n', X_test, '\n')
print ('y_Train \n \n', y_train, '\n')
print ('y_Test \n \n', y_test, '\n')


#Training Model Using Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred =spam_detect_model.predict(X_test)

#Comparing Predictions using confusion metrix

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

print ("Confusion Matrix = \n", confusion_m)

#Checking Accuracy Rate

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

print ("Accurance = ", accuracy)



