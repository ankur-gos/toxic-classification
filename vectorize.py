'''
vectorize.py
vectorize the word sequence
Ankur Goswami, agoswam3@ucsc.edu
'''
#%%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

trainf = 'data/train.csv'
testf = 'data/test.csv'

df = pd.read_csv(trainf)
documents = df['comment_text'].values.astype(str)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
tdf = pd.read_csv(testf)
test_documents = tdf['comment_text'].values.astype(str)
print(test_documents)
test_documents = vectorizer.transform(test_documents)
sentiments = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
df_final = pd.DataFrame()
df_final['id'] = tdf['id']
for sentiment in sentiments:
    y = df[sentiment].values
    clf = MultinomialNB().fit(X, y)
    predicted = clf.predict(test_documents)
    df_final[sentiment] = predicted
print(df_final)
