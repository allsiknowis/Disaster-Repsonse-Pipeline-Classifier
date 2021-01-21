# Disaster Response Pipeline Project

### Table of Contents
1. [Description](#description)

2. [Data](#data)

    i. [Dependencies](#dependencies)
        
    ii. [File Descriptions](#files)
        
3. [Instructions](#instructions)

4. [Acknowledgements](#acknowledgements)


### Description <a name="description"></a>
This project is part of Udacity's Data Scientist Nanodegree. The dataset comes from Figure Eight and contains labeled messages from real disaster situations. This project's purpose is to build a Natural Language Processing (NLP) machine learning model to categorize messages for help in real-time.

First, the data is processed using an ETL pipeline to extract the data from a .csv file which is then cleaned and saved to a SQLite database.

Next, a pipeline is used to train a machine learning model to classify text-based messages into several categories.

Finally, a web app is utilized to display the model's results as well as provide the ability to classify user-supplied messages.


### Data <a name="data"></a>

#### Dependencies <a name="dependencies"></a>
* Python: version 3.5+
* Data Analysis Libraries: numpy, pandas, re, sklearn
* File Saving Labrary: joblib
* Natural Language Processing Libraries: nltk
* SQLite Libraries: sqlalchemy
* Web App: flask, plotly

#### File Descriptions <a name="files"></a>
`ETL Preparation Notebook:` A full walkthrough of the ETL pipeline creation and implementation.

`ML Preparation Notebook:` A full walkthrough of the ML pipeline creation and implementation. It can also be used to train more machine learning models on the data.

`app/templates/*:` HTML template files for the web app

`app/run.py:` Launches the Flask web app that uses data from the ETL and ML pipelines to display relevent information and allow users to input custom messages for classification.

`data/process_data.py:` ETL pipeline used for extracting features, data cleaning and transforming, and loading the data into a SQLite database

`data/DisasterResponse.db:` The SQLite database created using process_data.py

`data/disaster_categories.csv:` The categories for the messages

`data/disaster_messages.cv:` The text-based messages

`models/train_classifier.py:` A machine learning pipeline for loading cleaned data, training the model, and saving the trained model as a .pkl file for future use.

`models/disaster_model.pkl:` the trained machine learning model


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:4000/


### Acknowledgements<a name="acknowledgements"></a>
* This program is part of [Udacity](https://www.udacity.com/)'s Data Scientist Nanodegree
* The data for this program comes from [Figure Eight](https://appen.com/figure-eight-is-now-appen/)
