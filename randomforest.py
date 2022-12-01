from read_data import *
from nlp import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

def print_baseline(X, y):
  dummy_clf = DummyClassifier(strategy="most_frequent")
  dummy_clf.fit(X, y)
  print(dummy_clf.score(X,y))

def get_cleaned_data(path):
  df = read_data(path)
  df['review_cleaned'] = df['review'].apply(lambda x: clean_text(x))
  return df

if __name__ == '__main__':
  # read and preprocess train dataset
  train = get_cleaned_data("../data/drugsComTrain_raw.tsv")
  # create Countvectorizer and tfidftransformer
  bow_transform = CountVectorizer(tokenize=lambda doc: doc, ngram_range=[1,1], lowercase=False)
  tfidf_transform = TfidfTransformer(norm=None)
  # train test split
  X = train[['review_cleaned','usefulCount']]
  y = train['rating']
  X_train, X_test, y_train, y_test = train_test_split(X,y)
  # train tfidftransformer and get tfidf matrix for X_train
  X_bow = bow_transform.fit_transform(X_train['review_cleaned'])
  X_tfidf = tfidf_transform.fit_transform(X_bow)
  # train random forest classifier
  rmf_clf = RandomForestClassifier(max_features=2000, class_weight="balanced")
  rmf_clf.fit(X_tfidf, y_train)
  # convert X_test to tfidf matrix
  X_bow_test = bow_transform.transform(X_test['review_cleaned'])
  X_tfidf_test = tfidf_transform.transform(X_bow_test)
  # predict on X_tfidf_test
  y_pred = rmf_clf.predict(X_tfidf_test)
  print(accuracy_score(y_test, y_pred))
  







