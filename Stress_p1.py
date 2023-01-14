# Importing all the necessary libraries

import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report


# Reading the data file into dataframe
df=pd.read_csv('stress.csv')

# Performing Exploratory Data Analysis inorder to check that our data has been read properly into the program
#df.head()
#df.tail()
#df.describe()
#df.isnull().sum()

# Performing Data-Preprocessing as we are dealing with Textual data
#import nltk
#import re
#from nltk.corpus import stopwords
#import string
nltk.download( 'stopwords' )
stemmer = nltk.SnowballStemmer("english")
stopword=set (stopwords.words( 'english' ))


def clean(text):
    text = str(text).lower()  
    text = re. sub('\[.*?\]',' ',text)  
    text = re. sub('https?://\S+/www\. \S+', ' ', text)
    text = re. sub('<. *?>+', ' ', text)
    text = re. sub(' [%s]' % re.escape(string. punctuation), ' ', text)
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)
    text = [word for word in text.split(' ') if word not in stopword]  
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text.split(' ') ]
    text = " ".join(text)
    return text
df [ "text"] = df["text"].apply(clean)




# As we are dealing with categorical data so inorder to convert that categorical data into numeric data Countvectorizer
# function is used.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array (df["text"])
y = np.array (df["label"])

cv = CountVectorizer ()
X = cv.fit_transform(x)
# print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)

  
# implementing Machine learning Algorithm

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)

# Performance
#y_pred = model.predict(xtest)
#cr = classification_report(ytest, y_pred)
#print(cr);


#Model accuracy ==> 78 %


user=input(" Enter your emotions of the day ")
data=cv.transform([user]).toarray()
output=model.predict(data)
if output == 1:
   print("You are having stress just Take Rest" )
else:
    print("You don't have any Stress Enjoy Your Day")



