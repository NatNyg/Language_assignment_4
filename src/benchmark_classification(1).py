"""
Start by importing the libraries I'll be using throughout the script 
"""

# system tools
import os
import argparse 
import sys

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#surpress warnings
import warnings
if not sys.warnoptions:
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"]="ignore" #also affect subprocesses
    
    
def benchmark_classification(dataset, features):
    """
This function loads and processes the user-defined dataset performing the following steps:
- read the data using pandas
- balancing the data using the predefined balance function from the utils folder 
- splitting the data into test and train 
- initialise the vectorizer and fit it on the training data to turn the documents into a vector of numbers instead of text - the vectorizer is then used on the testdata, but without fitting, becuase we want to test how well the training data works on the test data 
- get the feature names using the get_feature_names function
- initializing our classifier using the scikit learn LogisticRegression model to fit on the train data
- making predictions on the test data using the classifier 
- creating a metrics classification report using scikit learn 
- saving the report to the "out" folder 
    """
    filename = os.path.join("in",dataset)
    data = pd.read_csv(filename)
    data_balanced = clf.balance(data, 1000)
    X = data_balanced["text"]
    y= data_balanced["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.2, #create a 80/20 split 
                                                        random_state=42) 
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 lowercase = True, 
                                 max_df = 0.95, 
                                 min_df=0.05,
                                 max_features = features) 
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
     
    feature_names = vectorizer.get_feature_names()
    
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    
    y_pred = classifier.predict(X_test_feats)
    
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    file_path = 'out/bm_report.txt'
    sys.stdout = open(file_path, "w")
    print(classifier_metrics)
    sys.stdout.close()
                            
def parse_args():
    """
This function intialises the argument parser and defines the command line parameters
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ds","--dataset",required=True, help = "The dataset to make network analysis on")
    ap.add_argument("-fe","--features",required=True, help = "Maximum features (words) e.g. 100")
    args = vars(ap.parse_args())
    return args 

def main():
    """
The main function defines which function to run when the script is executed and what arguments given by the user should be passed to what functions 
    """
    args = parse_args()
    benchmark_classification(args["dataset"], int(args["features"]))
     
    
if __name__== "__main__":
    main()
