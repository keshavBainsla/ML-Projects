import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
data = pd.read_csv('C:/Users/kapil/Desktop/spam_or_not_spam.csv')
data.head()
email label
0 date wed NUMBER aug NUMBER NUMBER NUMBER NUMB... 0
1 martin a posted tassos papadopoulos the greek ... 0
2 man threatens explosion in moscow thursday aug... 0
3 klez the virus that won t die already the most... 0
4 in adding cream to spaghetti carbonara which ... 0
data.shape
(19, 2)
data.columns
Index(['email', 'label'], dtype='object')
data.drop_duplicates(inplace=True)
data.isnull().sum()
email 0
label 0
dtype: int64
#Tokenization (a list of tokens), will be used as the analyzer
#1.Remove punctuation
#2.Remove stopwords.
#3. Return list of clean text words
def process_text(text):
#1
nopunc = [char for char in text if char not in string.punctuation]
nopunc = ''.join(nopunc)

#2
clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#3
return clean_words
from sklearn.feature_extraction.text import CountVectorizer
msg = CountVectorizer(analyzer=process_text).fit_transform(data['email'])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(msg, data['label'], test_size = 0.20, random_state = 1)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression( random_state = 0).fit(X_train, Y_train)
C:\Users\Sahil\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
FutureWarning)
#Print the predictions
print(log_reg.predict(X_train))

#Print the actual values
print(Y_train.values)
[0 1 0 1 0 0 1 0 1 1 1 0 1 1 0]
[0 1 0 1 0 0 1 0 1 1 1 0 1 1 0]
#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = log_reg.predict(X_train)
print(classification_report(Y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(Y_train,pred))
print()
print('Accuracy: ', accuracy_score(Y_train,pred))
precision recall f1-score support

0 1.00 1.00 1.00 7
1 1.00 1.00 1.00 8

accuracy 1.00 15
macro avg 1.00 1.00 1.00 15
weighted avg 1.00 1.00 1.00 15

Confusion Matrix:
[[7 0]
[0 8]]

Accuracy: 1.0
#Print the predictions
print('Predicted value: ',log_reg.predict(X_test))

#Print Actual Label
print('Actual value: ',Y_test.values)
Predicted value: [1 1 0 1]
Actual value: [0 1 0 1]
#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = log_reg.predict(X_test)
print(classification_report(Y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(Y_test,pred))
print()
print('Accuracy: ', accuracy_score(Y_test,pred))
precision recall f1-score support

0 1.00 0.50 0.67 2
1 0.67 1.00 0.80 2

accuracy 0.75 4
macro avg 0.83 0.75 0.73 4
weighted avg 0.83 0.75 0.73 4

Confusion Matrix:
[[1 1]
[0 2]]

Accuracy: 0.75
