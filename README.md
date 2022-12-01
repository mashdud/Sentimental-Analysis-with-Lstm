# Drug Review Sentiment Analysis
Drugs.com is an online pharmaceutical encyclopedia that provides drug information for consumers and healthcare professionals. This analysis is based on drug reviews data from drugs.com. The goal of this analysis is to conduct sentiment analysis based on drug reviews by predicting customer ratings on specific drugs. 

## Dataset Description

The dataset is this analysis is publically available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29). The dataset provides 215K patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction. The attributes in the dataset include name of drug, name of condition, patient review, 10 star patient rating, date of review entry and number of users who found review useful.  

## Exploratory Data Analysis

The distribution of reviews is imbalanced as most of reviews were 8-10 and 1 stars. 

![ratingsdistribution](https://user-images.githubusercontent.com/26207455/116010209-a5a6c600-a5eb-11eb-82e5-ac4f16743cbd.png)


Most reviews were related to side effects. 

![frequent_words](https://user-images.githubusercontent.com/26207455/116010932-fcae9a00-a5ef-11eb-87c0-4ff458293660.png)


Number of reviews per year. 

![reviewsperyear](https://user-images.githubusercontent.com/26207455/116010733-dccaa680-a5ee-11eb-8f88-44483e1af988.png)

Top 20 drugs with 10/10 rating.

![top20drugswith10rating](https://user-images.githubusercontent.com/26207455/116010818-7003dc00-a5ef-11eb-82b3-96e77a5aa411.png)


Top 20 drugs with 1/10 rating.

![top20drugswith1rating](https://user-images.githubusercontent.com/26207455/116010847-91fd5e80-a5ef-11eb-8087-aac64418abc2.png)


## LSTM Modeling for Multi-class Classification (10 Classes)

Long Short Term Memory (LSTM) was used to predict ratings based on customer reviews. Before training LSTM, following steps were done to preprocess text reviews.
* Convert all text to lower case.
* Replace REPLACE_BY_SPACE_RE symbols by space in text.
* Remove symbols that are in BAD_SYMBOLS_RE from text.
* Remove stop words.
* Remove digits in text.

After text preprocessing, LSTM modeling were performed based on steps below.
* Vectorize customer reviews text by turning each text into either a sequence of integers or into a vector.
* Limit the data set to the top 10,000 words.
* Set the max number of words in each review at 200.  
* Train LSTM model.

![LSTM_summary](https://user-images.githubusercontent.com/26207455/116011098-24523200-a5f1-11eb-80cc-88df55e09c95.png)

The first layer is the embedded layer that uses 100 length vectors to represent each word. SpatialDropout1D performs variational dropout in NLP models. Two LSTM layers with 100 memory units. The output layer created 10 output values, representing 1-10 ratings. Activation function is softmax for multi-class classification. categorical_crossentropy is used as the loss function. 

![LSTM_losses](https://user-images.githubusercontent.com/26207455/116011445-1e5d5080-a5f3-11eb-9bc1-c46f77ebe8a2.png)

![LSTM_accuracy](https://user-images.githubusercontent.com/26207455/116011447-23220480-a5f3-11eb-8ef7-a94f643789b9.png)

The plots suggest that the model has a little over fitting problem, more data may help, but more epochs will not help using the current data. 

10 Classes Confusion Matrix: 

![LSTM_cm](https://user-images.githubusercontent.com/26207455/116011543-90359a00-a5f3-11eb-8b0c-cd9a053a0d84.png)
(horizontal: predicted labels, vertical: actual labels)

## Random Forest for Multi-class Classification (10 Classes)

### NLP pipelines for Text Preprocessing
* Convert all text to lower case.
* Replace contractions with longer forms.
* Remove special characters.
* Remove stopwords.
* Tokenization
* Lemmatization

### Feature Engineering
Since machine learning models do not accept the raw text as input data, we need to convert reviews into vectors of numbers. In this analysis, I've applied first bag of words and later convert it into Tf-Idf matrix. 

### Classification with Random Forest

![rmf_test_cm](https://user-images.githubusercontent.com/26207455/116012040-2074de80-a5f6-11eb-9b43-760b7efb5831.png)
(horizontal: predicted labels, vertical: actual labels)

## Model Performance

Random forest model based on tf-idf features has higher accuracy score than LSTM models.

Random Forest: 0.65

LSTM: 0.43

Both models performed better than baseline model (prediction based on most frequent ratings, accuracy score: 0.32). 


## Next Steps
* Fine tune both models to improve accuracy scores
* Try other modeling methods, bag of n-Grams, gradient boosting etc. 
* Try training models on larger data set on AWS to improve efficiency

## Refrence
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)
* [Multi-Class Text Classification with LSTM](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17)
* [Applying Text Classification Using Logistic Regression](https://medium.com/analytics-vidhya/applying-text-classification-using-logistic-regression-a-comparison-between-bow-and-tf-idf-1f1ed1b83640)
* [Multi-Class Text Classification with Keras and LSTM](https://djajafer.medium.com/multi-class-text-classification-with-keras-and-lstm-4c5525bef592)

