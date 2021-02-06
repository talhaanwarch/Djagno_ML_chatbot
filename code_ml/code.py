# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:37:34 2020

@author: TAC
"""


import nltk
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import random
import pandas as pd
import random
import json
with open('intents.json',encoding="utf8") as file:
    data = json.load(file)
    
df=pd.DataFrame(data['intents'])
df['tag']=df['tag'].str.lower()
#preprocess the dataframe
#lowering
df['patterns']=df['patterns'].apply(lambda x:[i.lower() for i in x])

#tokenization
from nltk.tokenize import word_tokenize
df['patterns']=df['patterns'].apply(lambda x:[word_tokenize(i) for i in x])


#lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))  

lemmatizer = WordNetLemmatizer() 
df['patterns']=df['patterns'].apply(lambda x:[[i for i in j if i not in stop_words] for j in x])
df['patterns']=df['patterns'].apply(lambda x:[[lemmatizer.lemmatize(i) for i in j] for j in x])


#remove noisy chracter
df['patterns']=df['patterns'].apply(lambda x:[[i for i in j if len(i)>2] for j in x])


df1=df.explode('patterns')
x=df1['patterns'].apply(lambda x:' '.join(x))
y=df1['tag']


#feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5))
X = vectorizer.fit_transform(x)

#classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#cross validation
from sklearn.model_selection import cross_val_score
print('accuracy is ',np.mean(cross_val_score(clf, X.toarray(), y, cv=3,n_jobs=-1)))

#save weights
import pickle
clf.fit(X.toarray(), y)

pickl = {
    'vectorizer': vectorizer,
    'classifier': clf
}
pickle.dump( pickl, open( 'models' + ".p", "wb" ) )



