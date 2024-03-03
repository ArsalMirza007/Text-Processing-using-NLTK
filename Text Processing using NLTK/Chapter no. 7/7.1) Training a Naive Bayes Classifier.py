# Import necessary modules and functions
import nltk
from nltk.corpus import movie_reviews
from nltk import FreqDist, classify, NaiveBayesClassifier

# Download the movie_reviews dataset if not already downloaded
nltk.download('movie_reviews')

# Check the categories in the Movie Reviews dataset
print(movie_reviews.categories())  # Output: ['neg', 'pos']

# Create labeled feature sets from the Movie Reviews dataset
def bag_of_words(words):
    return dict([(word, True) for word in words])

labeled_features = []
for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        words = movie_reviews.words(file_id)
        labeled_features.append((bag_of_words(words), category))

# Split the labeled feature sets into training and testing sets
split_ratio = 0.8
split = int(len(labeled_features) * split_ratio)
train_feats = labeled_features[:split]
test_feats = labeled_features[split:]

# Train a Naive Bayes classifier on the training features
nb_classifier = NaiveBayesClassifier.train(train_feats)

# Check the labels of the classifier
print(nb_classifier.labels())  # Output: ['neg', 'pos']

# Create bag of words features for a negative review
negfeat = bag_of_words(['the', 'plot', 'was', 'ludicrous'])

# Classify the negative review using the trained classifier
print(nb_classifier.classify(negfeat))  # Output: 'neg'

# Create bag of words features for a positive review
posfeat = bag_of_words(['kate', 'winslet', 'is', 'accessible'])

# Classify the positive review using the trained classifier
print(nb_classifier.classify(posfeat))  # Output: 'pos'

# Calculate the accuracy of the classifier on the test set
accuracy_score = classify.accuracy(nb_classifier, test_feats)
print(accuracy_score)  # Output: Accuracy score

# Get probability distributions for a test feature
probs = nb_classifier.prob_classify(test_feats[0][0])

# Check the class labels in the probability distribution
print(probs.samples())  # Output: dict_keys(['neg', 'pos'])

# Get the class with the highest probability
print(probs.max())  # Output: 'pos'

# Get the probability of the 'pos' class for the test feature
print(probs.prob('pos'))  # Output: Probability

# Get the probability of the 'neg' class for the test feature
print(probs.prob('neg'))  # Output: Probability

# Get the most informative features of the classifier
print(nb_classifier.most_informative_features(n=5))

# Train a Naive Bayes classifier with Laplace smoothing
nb_classifier = NaiveBayesClassifier.train(train_feats)

# Calculate the accuracy of the Laplace smoothed classifier on the test set
accuracy_score = classify.accuracy(nb_classifier, test_feats)
print(accuracy_score)  # Output: Accuracy score

