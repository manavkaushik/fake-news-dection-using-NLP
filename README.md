# Fake News Dection Using NLP

## About the Code:

The code has been built keeping in reference to the paper:
https://aclweb.org/anthology/W18-5513 .
The first part of the code tries to mimic some part of the same paper i.e. using Logistic
Regression and BiLSTM models for creating an appropriate classifier. Following
pre-processes were implemented before running the model:

  ● Data Cleaning by lowercasing all texts and dropping rows with NULL variables.
  
  ● Using Count Vectorizer for transforming the training and testing data.
  
  ● Using tf-idf Vectorizer for transforming the training and testing data.
  
  ● Removing all stop words (using nltk library).
  
  ● Text Stemming
  
  
The second part of the code yields the best classifier: XGBoost (a Boosting Ensemble).
This part of the code uses similar kinds of pre-processes as the first part but using a pipeline
with a few helper functions, sentiment analysis and ‘Lemma tokenization’.


## Results achieved:

● Six-Way Classification: Best Accuracy Score (F1 Score) using XGBoost: 0.40 or
40%

● Binary Classification: Best Accuracy Score (F1 Score) using XGBoost: 0.72 or
72%



Thus, the best classifier turns out to be: XGBoost (with tuned hyperparameters). The
classification results turned out to be even better than those in the journal paper (where the
best results were: 0.38 for Six Way and 0.70 for Binary classification)


## How I achieved this result?

The exact processes that I followed to achieve this result are as follows:

  ● Data Preprocessing:
  
    ○ Removing irrelevant columns like index and JSON id.
    ○ By removing data points having Null values.
    ○ Noun & Adjective Tagging.
    ○ Vectorization using Lemma Tokenizer.
    ○ Segregating text columns and numeric columns to perform further
      transformations.
    ○ Calculating sentiments of the statements through Lexicon (using ‘Empath’
    library).
    
  ● Building pipelines to put all these things together.
  
  ● Performing a GridSearch for tuning XGBoost Hyperparameters.
  
  ● Ran the model on the transformed data to achieve 39% accuracy.
  
  
## Why XGBoost?

  ● XGBoost is a regularized boosting technique which helps reducing overfitting.
  
  ● Unlike other tree algorithms, XGBoost make splits up to the max_depth specified
    and then start pruning the tree backwards and remove splits beyond which there is
    no positive gain.
    
  ● XGBoost is a boosting algorithm i.e. it tries to fit the data by using multiple simpler
    models, or so-called base learner/weak learner.
    
  ● It is invariant under monotone transformations of the inputs.
  
  ● It performs implicit variable selection.
  
  ● It can capture non-linear relationships in the data.
  
  ● It can capture high-order interactions between inputs.
  
  
## Different Ideas tried:

  ● Firstly, I tried implementing a very simple model with only ‘statement’ (or ‘news’)
    column as an input parameter and LogisticRegression as the classifier. The accuracy
    achieved was exactly similar to that in the paper i.e. 0.25.
    
  ● Surprisingly, I couldn’t achieve significantly higher results even after adding the
    ‘justification’ columns and metadata.
    
  ● After this, I implemented some other well-known algorithms like SVM, AdaBoost,
    Random Forest and NaiveBayes (using sklearn) and a bi-directional LSTM and their
    stackings (using Keras), but none of them which couldn’t achieve accuracy better
    than Logistic regression.
    
  ● Then, I implemented an XGBoost algorithm with all the above-mentioned
    modifications (some of the ideas were drawn from kaggle competitions and blogs)
    which gave an accuracy of 39%.
    
    
## Additional Notes:

- I also tried creating a very strong ensemble (but couldn’t complete it due to time
constraints) of a Feedforward Neural Network (1 Input + 2 HIdden + 1 Output layer)
and an XGBoost. The essence is to use FNN as a feature extraction algorithm by
taking the pre-final layer (i.e the last hidden layer) as input for XGBoost. The idea
behind this ensemble is to leverage the strongest abilities of both these algorithms:
complex relation-learning of FNN and the boosting property of XGBoost while at the
same time avoiding overfitting.

- In a nutshell, there are several techniques to solve the problem of Text Classification
which can be effectively implemented with adequate ML and data expertise and sound
research temperament.


## References:

  ● Libraries used:
  
    ○ Numpy
    ○ Pandas
    ○ Matplotlib
    ○ Seaborne
    ○ Sklearn
    ○ XGBoost
    ○ NLTK
    ○ Unicode
    ○ String
    ○ Tensorflow
    ○ Keras
    ○ Empath (for emotion analysis)
    ○ Textblob (for sentiment analysis)
    
  ● Papers cited :
  
    ○ https://aclweb.org/anthology/W18-5513
    ○ https://arxiv.org/pdf/1602.06979.pdf
    ○ https://www.aclweb.org/anthology/E17-1104
    ○ https://wsdm-cup-2018.kkbox.events/pdf/8_Ensembling_XGBoost_and_Neural_
      Network_for_Churn_Prediction_with_Relabeling_and_Data_Augmentation.pdf
    ○ https://www.aclweb.org/anthology/O18-1021
    ○ https://www.ijcai.org/Proceedings/16/Papers/408.pdf
    
  ● Websites cited:
  
    ○ https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-underst
      and-and-implement-text-classification-in-python/
    ○ https://medium.com/@himanshu_23732/sentiment-analysis-with-textblob-6bc2eb
      9ec4ab
    ○ https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltktf-
      idf-xgboost-and-more-b83451a327e0
    ○ https://www.kaggle.com/diveki/classification-with-nlp-xgboost-and-pipelines
    ○ https://towardsdatascience.com/machine-learning-nlp-text-classification-using-sci
      kit-learn-python-and-nltk-c52b92a7c73a
    ○ https://stackabuse.com/text-classification-with-python-and-scikit-learn/
