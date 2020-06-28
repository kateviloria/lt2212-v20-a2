import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)

# My Own Imports
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import string
import collections
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import classification_report 
        # --> Used at first, excessive info and took more time to process


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X


def tokenize_text(data):
    """
    Tokenizes text and counts frequencies.

    Args:
        samples - 20 newsgroup data
    
    Returns:
         A list of dictionaries. Each dictionary represents a document. Within each dictionary, the key is the
        word in the document and its value is the frequency.
        [ { word : 4, another_word : 2, a_word : 0}, { word : 0, a_diff_word : 1, what_word : 1}]

        AND

        A list of all the words that appear in the data. Words only appear once.
    """

    # from NLTK : words that search engines have been programmed to ignore (ie. pronouns, 'during', 'very')
    stop_words = stopwords.words('english')

    # news_data = newsgroups_all.data

    # for trial BABY DATA 
        # baby_data = news_data[:21]
        # data_text = baby_data
    
    # tokenize data 
    text_list = [] # should be list of lists, each inner list is text of a file
    master_list = [] # list of words -> columns for array (will need for indexing)
    for each_file in data: # CHANGE TO BABY_DATA for trial; NEWS_DATA for actual
        word_list = []
        # make string into list
        string_to_list = each_file.split()
        for every_word in string_to_list:
            if every_word.isalpha(): # filters out integers and punctuation
                no_capitals = every_word.lower() # lowercases all characters of word
                if no_capitals not in stop_words: # filters through NLTK stop words list
                    word_list.append(no_capitals)
                    if no_capitals not in master_list:
                        master_list.append(no_capitals)
        text_list.append(word_list)

    # count freq
    word_freq = [] # list of dictionaries
    for every_file in text_list:
        article_dict = {} # { WORD : FREQUENCY }, dictionary per text
        # goes through list per file and uses counter to create a dictionary
        article_dict = collections.Counter(every_file) 
        word_freq.append(article_dict)

    return word_freq, master_list


def extract_features(samples):
    """
    Takes tokenized text and creates a numpy array. 
    Filters out words (deletes columns within the array) that appear less than 20 times in the data.

    Args:
        samples - 20 newsgroup data

    Returns:
        A numpy array of word occurrences.
            rows = documents
            columns = words
   """

    print("Extracting features ...")

    # calls tokenize text to get word_freq and master_list
    word_freq, master_list = tokenize_text(samples)
    print('master_list, word_freq done')
    
    # number of rows -> docs
    doc_rows = len(word_freq)
    print('have doc rows')

    # number of columns -> words in entire lexicon
    word_columns = len(master_list)
    print('have word columns')
    
    # make array with 0's for all the words that doesn't appear in doc
    final_array = np.zeros((doc_rows, word_columns))
    print('initial final array w all zeroes done')

    # to move to next doc
    row_index = 0
    
    for every_article in word_freq:
        for every_word in every_article.keys():
            word_index = master_list.index(every_word) # get index from master_list
            word_count = every_article[every_word] # get word count from dictionary
            final_array[row_index, word_index] = word_count # put count in correct index within array
        row_index += 1 # next doc/row

    print('final_array done')
    # print(final_array)
   
    # CHECK SHAPE OF FIRST VEC
        # shape_of_vec = final_array.shape
        # print(shape_of_vec)

    # reduce by total word counts per column
    array_column_sum = np.sum(final_array, axis = 0) # sum of each word in all documents
    # filtered will only have words that appear more than 20 times within entire data set
    array_filter = array_column_sum > 20 
    print('about to filter final array')
    filtered_array = final_array[:, array_filter] # [ROW, COLUMN]
    # shape_filtered = filtered_array.shape

    # consider words that appear in all documents (ie. subject)
        # tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
        # tfidf_array = tfidf.fit_transform(reduced_array)
        # shape_tfidf = tfidf_array.shape
        # print(shape_tfidf)
        # SEEMED TO NOT RETURN AN ARRAY ACCORDING TO ASSERT FUNCTION

    print('Done extracting features')
    
    return filtered_array


##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    
    # reduce dimension using Truncated SVD (Singular Value Decomposition)
    svd = TruncatedSVD(n)
    svd_array = svd.fit_transform(X)

    return svd_array


##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = KNeighborsClassifier(n_neighbors=5) # default: weights=uniform, algorithm=auto
    elif clf_id == 2:
        clf = DecisionTreeClassifier()
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf


#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80) # default: shuffle = True
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X, y)


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    # y is y_true
    predicted = clf.predict(X)
    accuracy = accuracy_score(y, predicted)
    precision = precision_score(y, predicted, average='weighted')
    recall = recall_score(y, predicted, average='weighted')
    f1 =  f1_score(y, predicted, average='weighted')
    # calculates precision, recall, F-measure, support (extra)
        # class_report = classification_report(y, clf.predict(X))
        # print(class_report) -> took too long
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F-measure: ', f1)
    

######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )

