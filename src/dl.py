"""
Start by importing the libraries I am using for this script 
"""
import os
import sys
import argparse


# simple text processing tools
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

# data wrangling
import pandas as pd
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


# scikit-learn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


"""
These three first functions (strip_html_tags, remove_accented_chars and pre_process_corpus) are functions that helps preprocess the corpus we'll be performing deep learning tasks on. Basically they help normalise the docs so the text is clean and ready to work with. 
"""

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
    
    return norm_docs

def load_and_process_data(data_set):
    """
This function loads the user-defined dataset and processes it by performing the following steps:
- read data using pandas
- replace the lables 0 and 1 with non-toxic and toxic 
- fetch the text and label from the dataset 
- split the dataset into test and train datasets
- normalise the data using the pre_process_corpus function 
- tokenize the documents using the tensorflow keras tokenizer
- set the padding value to 0 (adding 0's to the end of documents that are shorter than others, ensuring that documents have the same length
- turn the text into sequences 
- set maximum sequence length to 1000, ensuring a max of 1000 tokens in each document 
- add padding sequences to test and train data
- using sci-kit learn's labelbinarizer to binarize the labels 
    """
    filename = os.path.join("in","toxic",data_set)
    dataset = pd.read_csv(filename)
    dataset.replace({0:"non-toxic", 1:"toxic"}, inplace=True)
    text = dataset["text"].values
    label = dataset["label"].values
    X_train, X_test, y_train, y_test = train_test_split(text,
                                                        label, 
                                                        test_size = 0.2, 
                                                        random_state = 42) 
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    t = Tokenizer(oov_token = '<UNK>')

    t.fit_on_texts(X_train_norm) 

    t.word_index["<PAD>"] = 0
     
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    MAX_SEQUENCE_LENGTH = 1000

    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train_pad, y_train, X_test_pad, y_test, t, MAX_SEQUENCE_LENGTH
    
def define_model(embedding_size, t, MAX_SEQUENCE_LENGTH):
    """
This function defines the model we want to use for the text classification task by doing the following:
- clearing session for models and parameters
- define parameters for model (overall vocabulary size, embedding size, number of epochs and batch size)
- create the model using the Sequential tensorflow keras model 
- adding layers (embedding, convolution, pooling and fully connected classification). We'll be using sigmoid and binary crossentropy since we're dealing with a binary classification problem (toxic or non-toxic)
    """
    tf.keras.backend.clear_session()
    
    VOCAB_SIZE = len(t.word_index)
    EMBED_SIZE = embedding_size
    EPOCH_SIZE = 2 
    BATCH_SIZE = 128

    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                     kernel_size=4, 
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model, EPOCH_SIZE, BATCH_SIZE

def train_and_evaluate_model(X_train_pad, y_train, X_test_pad, y_test, model, EPOCH_SIZE, BATCH_SIZE):
    """
This function trains and evaluates the model just defined by doing the following:
- fitting the model on the train data and saving the history of the training
- evaluating the model using the evaluate function on the test data and saving the scores 
- making predictions using the predict function and saving the results
- assigning the labels 
- making a classification report using scikit learn's function, and saving the report to the "out" folder. 
    """
    history = model.fit(X_train_pad, y_train,
                        epochs = EPOCH_SIZE,
                        batch_size = BATCH_SIZE,
                        validation_split = 0.1, 
                        verbose = True) 
    scores = model.evaluate(X_test_pad, y_test, verbose = 1)
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    file_path = 'out/dl_report.txt'
    sys.stdout = open(file_path, "w")
    print(classification_report(y_test, predictions, target_names = ['non-toxic', 'toxic']))
    sys.stdout.close()

    
def parse_args():
    """
This function intialises the argumentparser and adds the command line parameters 
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ds","--dataset",required=True, help = "The dataset to perform text classification")
    ap.add_argument("-es","--embedding_size",required=True, help = "Desired embedding size, e.g. 300")
    args = vars(ap.parse_args())
    return args 

def main():
    """
The main function defines which functions to run when the script is executed, and which command line parameters should be passed to what functions. 
    """
    args = parse_args()
    X_train_pad, y_train, X_test_pad, y_test, t, MAX_SEQUENCE_LENGTH = load_and_process_data(args["dataset"])
    model, EPOCH_SIZE, BATCH_SIZE = define_model(int(args["embedding_size"]), t, MAX_SEQUENCE_LENGTH)
    train_and_evaluate_model(X_train_pad, y_train, X_test_pad, y_test, model, EPOCH_SIZE, BATCH_SIZE)
     
    

if __name__== "__main__":
    main()
   





