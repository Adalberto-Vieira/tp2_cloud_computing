from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd

df = pd.read_csv('training.csv', header=0, sep=";")

df['country_code'] = df.country_code.map({'US': 1}).fillna(0).astype(int)

text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])

text_clf.fit(df["text"], df["country_code"])

df2 = pd.read_csv('test.csv', header=0, sep=";")
df2['country_code'] = df2.country_code.map({'US': 1}).fillna(0).astype(int)

predicted = text_clf.predict(df2["text"])

pickle.dump(text_clf, open("../flask_app/app/model.pickle", 'wb'))

