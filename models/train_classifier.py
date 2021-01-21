# import packages
import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

nltk.download(['punkt','stopwords','wordnet'])


def load_data(data_file):
    '''
    Description:
        Loads the supplied data file, cleans the data using etl_pipeline.py
        which saves the data to a sqlite table, loads the data from that sqlite
        table, and defines the features and labels arrays.

    Args:
        data_file - file path - contains the .csv files to be
            cleaned and then used for modeling.

    Returns:
        X - pandas dataframe - features data
        Y - pandas dataframe - classification labels
    '''

    print('Loading database...')

    # load to database
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('labeled_messages', engine)

    print('Database loaded!\n')

    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)

    return X, Y

def tokenize(text):
    '''
    Convert text to all lowercase characters,
    tokenize text, and stem and lemmatize text.

    Args:
    text - string - the message sent for processing

    Returns:
    filtered_sentence - list of strings - a list of the cleaned and stemmed message words
    '''

    # convert all text to lowercase characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    word_tokens = word_tokenize(text)

    # stem and lemmatize non-stop words to reduce redundancy
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_sentence = [stemmer.stem(word) for word in word_tokens if word not in stop_words]
    filtered_sentence = [lemmatizer.lemmatize(word, pos='v') for word in stemmed_sentence]

    return filtered_sentence

def build_model(X, y):
    '''
    Description:
        Builds a machine learning pipeline that comprises CountVectorizer,
        TfidfTransformer, and MultiOutputClassifier.

    Args:
        X - pandas dataframe - features data
        y - pandas dataframe - classification labels
        model - classifier - the classifier to be used in building
            the pipeline and to be tuned with GridSearchCV.
        params - dictionary -  the hyperparamters for the given model
            to be tuned using GridSearchCV.

    Returns:
        model_pipeline - classifier - the tuned machine learning model.

    '''

    print('Splitting train and test data...')

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Building model...')

    # text processing and model pipeline
    clf = AdaBoostClassifier(random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf))
        ])

    print('Fitting pipeline...')

    pipeline.fit(X_train, y_train)

    print('Pipeline fitting complete!')

    # define parameters for GridSearchCV
    parameters = {'vect__min_df': [1, 2, 3],
                  'tfidf__smooth_idf': [True, False],
                  'clf__estimator__n_estimators':list(range(10, 100, 10)),
                  'clf__estimator__learning_rate': [0.001, 0.01, 0.1, 1.0]
                  }

    print('Tuning hyperparameters...')

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, parameters)

    print('Training tuned model...\n')

    cv = train(X_train, X_test, y_train, y_test, cv)

    model_pipeline = cv.best_estimator_

    print('Best model found!\n')

    return model_pipeline


def train(X_train, X_test, y_train, y_test, model):
    '''
    Description:
        Splits the supplied data into training and testing sets,
        fits the training data to the supplied model, makes predictions
        on the test data, and prints out the test results

    Args:
        X - array - features array
        y - array - labels array
        model - classifier - the classifier to be used in building
            the pipeline and to be tuned with GridSearchCV.

    Returns:
        model - classifier - the trained classifier that
            is supplied to the function as 'model'

    '''

    print('Fitting training data to model...')

    # fit model
    model.fit(X_train, y_train);

    print('Predicting on test data...\n')

    # predict on train and test data
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    # output model train results
    for i, column in enumerate(y_train):
        print(column)
        print(classification_report(y_train[column], y_train_preds[:, i]))

    # output model test results
    for i, column in enumerate(y_test):
        print(column)
        print(classification_report(y_test[column], y_test_preds[:, i]))

    print('Training complete!\n')

    return model


def export_model(model, model_filepath):
    '''
    Description:
        Exports the supplied trained model to a .pkl file.

    Args:
        model - classifier - the trained classifier

    Returns:
        None

    '''

    print('Exporting model...')

    # Export model as a pickle file
    joblib.dump(model, model_filepath)

    print('Export complete!\n')



def run_pipeline(data_file, model_filepath):
    '''
    Description:
        Runs the whole machine learning pipeline by calling the
        relevant functions.

    Args:
        data_file - filepath - the filepath for the .csv file to
        be used in the machine learning pipeline.

    Returns:
        None

    '''
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model(X, y)  # build model pipeline
    export_model(model, model_filepath)  # save model

def main():
    if len(sys.argv) == 3:
        data_file, model_filepath = sys.argv[1:] # get filename of dataset and the .pkl file
        run_pipeline(data_file, model_filepath)  # run data pipeline

    else:
        print('Please provide a filepath for the disaster messages database '\
              'as the first argument and the filepath for the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
