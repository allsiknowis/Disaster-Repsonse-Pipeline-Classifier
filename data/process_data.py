# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def read_file(messages_filepath, categories_filepath):
    '''
    Description:
        Reads in .csv files to variables for messages and categories.

    Args:
        file_path - file path - contains the .csv files to be
            cleaned and then used for modeling.

    Returns:
        messages - pandas dataframe - contains the disaster messages
        categories - pandas dataframe - contains the categories for the disaster messages
    '''

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    print('Imports successful!')

    return messages, categories

def clean_data(messages, categories):
    '''
    Description:
        Cleans the data in the 2 dataframes supplied to the function by transforming
        'categories' columns and values, merging 'messages' and 'categories' dataframes,
        removing duplicates, dropping NaNs, and making sure all values are 1s and 0s.

    Args:
        messages - pandas dataframe - contains the disaster messages
        categories - pandas dataframe - contains the categories for the disaster messages

    Returns:
        df - pandas dataframe - cleaned dataframe containing the combined data of
            'messages' and 'categories'.
    '''

    print('\nCleaning data...')

    # merge the datasets
    df = pd.merge(messages, categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.head(1)

    # use 'row' to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda i: i[:-2]).iloc[0, :].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    # remove 2s from the data and replace with 0s
    categories.replace(2, 0, inplace=True)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)

    # drop missing values
    df.dropna(subset=category_colnames, inplace=True)

    print('Data cleaned successfully!\n')

    return df

def save_sql_db(df, database_filepath):
    '''
    Description:
        Saves the supplied dataframe as a sqlite table.

    Args:
        df - pandas dataframe - dataframe containing the data to be saved
           as a sqlite table.
    Returns:
        None

    '''

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')
    engine.dispose()
    print('Database created successfully!\n')

def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        messages, categories = read_file(messages_filepath, categories_filepath)

        df = clean_data(messages, categories)

        save_sql_db(df, database_filepath)

    else:
        print('Please provide filepaths for the messages and categories '\
              'datasets as the first and second arguments, and '\
              'the filepath of the cleaned data database '\
              'as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == "__main__":
    main()
