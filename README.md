# Langugage_assignment_4 - Text classification

## This is the repository for assignment 4 in Language Analytics.

### Project Description
This project performs text classification using two different approaches: benchmark classification and deep learning. The goal is for both of the approaches to be able to predict whether a comment from our dataset "VideoCommentsThreatCorpus.csv" is toxic or not. The aim is then to compare the two different methods, in order to see the difference in the results and thus conclude which of the methods that are performing better on this specific classification task. 

### Repository Structure

The repository includes three folders:

- in: this folder should contain the data that the code is run on
- out: this folder will contain the results after the code has been run
- src: this folder contains the script of code that has to be run to achieve the results
    
### Method
As mentioned this project contains two scripts, which means I have essentially used two different methods. The first one is benchmark classification, and for this script the following things are done:
- I first load the data using pandas
- I then balance the data using the clf function from the utils folder and splits the data into train/test datasets 
- I then initialise the TfidfVectorizer to get the weighted values, and firstly fits the vectorizer on my traindata to turn all of our documents into a vector of numbers, instead of text. I then use the vectorizer, doing the same thing on the test data - but without fitting it, since we want to test how well the training works on the test data. 
- I then use the vectorizer to get the feature names using the get_feature_names() function. 
- I then initialise and fit my classifier on the training data, finding correlations between features and labels (using LogisticRegression from sci-kit learn)
- I then use the classifier to make predictions on the test-data using the predict function 
- I then create a metrix for the classification report using the classification_report function. 
- Finaly I save the classification to the "out" folder. 

The second method is deep learning, and for this script the following things are done:
- I firstly include three helping functions: strip_html_tags, remove_accented_chars and pre_process_corpus. These functions helps me to preprocess my corpus for text classification, using deep learning 
- I then read and split my dataset, but first changing the "0" and "1" into my preferred labels; "non-toxic" and "toxic". 
- I then use my pre_process_corpus on my train and test data 
- I then initialise a tokenizer and fits it on my documents in the training data. After this a padding value is set in order to ensure that the documents have the same length. If a document is shorter, we just pad by adding 0's to the end of the document
- I then convert my texts to sequences, set the maximum number of tokens in each document to 1000 and add the padding sequences. 
- I then use my label_binarizer on my labels, in order to get 0 and 1 to work with again 
- I then clear the session for previous models, define the parameters for the new model, create the model (Sequential) and add layers. 
- I then fit/train the model on the training data and evaluate the model
- I then make predictions on the test data, including that I want a 0.5 decission boundary for my predictions 
- Lastly I assign the desired labels (non-toxic/toxic) and print my classification report to the "out" folder. 




### Usage

In order to reproduce the results I have gotten (and which can be found in the "out" folder), a few steps has to be followed:

1) Install the relevant packages - relevant packages for both scripts can be found in the "requirements.txt" file.
2) Make sure to place the script in the "src" folder and the data in the "in" folder. The data used for this project can is placed in the in folder.
3) Run the script from the terminal and remember to pass the required arguments:
-> For the dl.py script: -ds (dataset) and -es (embedding_size) 
-> For the benchmark_classification.py script: -ds (dataset) and -fe (max_features) 
-> Make sure to navigate to the main folder before executing the script - then you just have to type the following in the terminal: 

"python src/dl_ass.py -ds {name_of_the_dataset} -es {embedding_size}" for the dl.py
"python src/benchmark_classification.py -ds {name_of_the_dataset} -fe {max_features}"


This should give you the same results as I have gotten in the "out" folder.

### Results

As the aim of this project was to compare the two classification methods (benchmark classification and deep learning), it is interesting to look into the results of my classification reports. This shows what I would have expected before going into the tasks; that the deep learning method performs better on this task. With a max precision of 98% for predictions on the non-toxic comments, this model performs 18% better than the benchmark classification does at best, with a maximum precision of 80% for predictions on the toxic comments. However, I do actually think the results are pretty good for both of the methods, and the benchmark classification is extremely fast to run, which is something worth taking into consideration.
