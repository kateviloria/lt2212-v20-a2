# LT2212 V20 Assignment 2

Put any documentation here including any answers to the questions in the assignment on Canvas.

__Part 1 - creating the feature table__  
Tokenized the text by:  
    - turning text file into a list of strings (each word is a string)   
    - filtering out all integers and punctuation  
    - putting all characters in lowercase  
    - filter out any words that are stopwords (via NLTK)  
    --> returns a list of lists

Using that list of lists:  
    - create a dictionary for each text file where the key is the word and the number of occurrences of the word within the text is the value  
    --> returns a list of dictionaries [ {a_word : 3, other_word : 6 }, { a_word : 1, another_word : 4]}

Create numpy array:  
    - initial array has all zeroes   
    - insert word counts by using index of master_list (list with all of the possible words in the data) and a row index counter (for each document)  
    - only keep columns (which are words) that appear at least 20 times in the data (to make data smaller)

__Part 2 - dimensionality reduction__  
Reduced dimensions by using:  
    a2.py - Truncated Singular Value Decomposition (SVD)  
    a2BONUS.py - Principal Component Analysis (PCA)  

__Part 3 - classify and evaluate__  
Model 1 - K Nearest Neighbors Classifier  
Model 2 - Decision Tree Classifier  

__Part 4 - try and discuss__  
*Unreduced*  
(9718 dimensions)
|               | Accuracy | Precision | Recall | F-measure |
|---------------|----------|-----------|--------|-----------|
| K-Neighbors   | 0.41     | 0.65      | 0.41   | 0.46      |
| Decision Tree | 0.63     | 0.63      | 0.63   | 0.63      |

*Reduced via Truncated SVD*  
K-Neighbors
|            | Accuracy | Precision | Recall | F-measure |
|------------|----------|-----------|--------|-----------|
| 50% (4859) | 0.44     | 0.65      | 0.44   | 0.49      |
| 25% (2429) | 0.47     | 0.61      | 0.47   | 0.50      |
| 10% (971)  | 0.51     | 0.60      | 0.51   | 0.53      |
| 5% (485)   | 0.49     | 0.55      | 0.49   | 0.50      |

Decision Tree
|            | Accuracy | Precision | Recall | F-measure |
|------------|----------|-----------|--------|-----------|
| 50% (4859) | 0.30     | 0.30      | 0.30   | 0.30      |
| 25% (2429) | 0.32     | 0.32      | 0.32   | 0.32      |
| 10% (971)  | 0.33     | 0.33      | 0.33   | 0.33      |
| 5% (485)   | 0.34     | 0.34      | 0.34   | 0.34      |

__*Observations*__  
For K-Neighbors:  
    The accuracy appears to improve (at the highest, a 10% improvement) when the dimensions are reduced. However, the reduction from 10% to 5% of the original dimensions seems to show some sort of parabolic peak in a sense that there is a point where the model reaches a high point of accuracy and begins to decrease. Its precision continues to decrease as the dimensions are reduced to a greater scale. This is quite interesting since it means that there is an increase in false positives. In contrast, the recall appears to have a gradual increase when we reduce the dimensions more. This is understandable since precision and recall are connected where if one improves, the other typically reduces. Similar to accuracy, the F-measure gradually increases when dimensions are reduced but then decreases when the dimensionality reduction goes from 10% to 5%.

For Decision Tree:  
    It was surprising to see that the outcomes for all four evaluation measurements were consistent for the model that was not reduced. It was also very interesting to see the ~50% decrease of all measurements after dimensionality reduction. Regardless of how much the dimensions were reduced, all the results were within the range of 30-34% which was not what I expected given the general relationships the four measurements have (espcially precision and recall) with each other.

To compare:  
    I think the Decision Tree classifier being used in the unreduced model gave the best results since it makes sorting decisions by breaking down the data into smaller subsets and creating a classification model. To me it makes sense that the less dimensions there were, the more it was unable to correctly classify. In comparison with K-Nearest Neighbor, it was still able to retain (and to a degree, improve) features that would help with classification when the dimensions were reduced. This is probably because its implementation is based on finding the smallest distance between the training points and the testing point and choosing the class of those training points.


__Part Bonus - another dimensionality reduction__  
*Reduced via PCA*  
K-Neighbors
|            | Accuracy | Precision | Recall | F-measure |
|------------|----------|-----------|--------|-----------|
| 50% (4859) | 0.44     | 0.64      | 0.44   | 0.49      |
| 25% (2429) | 0.50     | 0.62      | 0.50   | 0.53      |
| 10% (971)  | 0.50     | 0.58      | 0.50   | 0.51      |
| 5% (485)   | 0.52     | 0.57      | 0.52   | 0.53      |

Decision Tree
|            | Accuracy | Precision | Recall | F-measure |
|------------|----------|-----------|--------|-----------|
| 50% (4859) | 0.317    | 0.320     | 0.317  | 0.318     |
| 25% (2429) | 0.310    | 0.309     | 0.310  | 0.310     |
| 10% (971)  | 0.324    | 0.323     | 0.324  | 0.324     |
| 5% (485)   | 0.320    | 0.317     | 0.320  | 0.318     |

__*Observations*__  
For K-Neighbors:  
    The patterns are quite similar to when I used the Truncated SVD to reduce the dimensions. However, there seems to be an increase in all measurements even from 10% to 5% reduction instead of a decrease seen in Part 4. The improvements are not big but they are still improvements.

For Decision Tree:  
    The patterns are the same as when Truncated SVD was used. There is a ~50% decrease within all measurements. I added another integer for this table because the changes within the measurements cannot be seen from two decimal places. Similar to the table for Decision Tree in Part 4, the results seem to all range from 31% to 32.4%. To me, the appear to be all over the place and I would need to run more tests in order to identify a pattern.

To compare (Truncated SVD vs PCA):  
    It makes sense that the effects of both Truncated SVD and PCA to both classifiers are similar. While SVD is able to analyse the data into independent components, PCA does the same and disregards less significant components. Since I used Truncated SVD, it is basically doing what PCA was meant to do--truncating the less important basis vectors from the original SVD matrix. 