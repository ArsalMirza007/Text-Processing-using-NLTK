import nltk
from nltk.corpus import movie_reviews
from nltk.metrics import accuracy
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import precision_score, recall_score
from nltk.probability import FreqDist
from nltk.metrics import BigramAssocMeasures

# Define a function to calculate high information words
def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    word_scores = {}
    label_word_freq = nltk.FreqDist()
    total_word_freq = nltk.FreqDist()

    for label, words in labelled_words:
        label_word_freq[label] += 1
        total_word_freq.update(words)

    for label, words in labelled_words:
        for word in words:
            if len(word) > 2:
                word_scores[word] = word_scores.get(word, 0) + score_fn(label_word_freq[label, word],
                                                                       (total_word_freq[word] - label_word_freq[label, word]),
                                                                       label_word_freq[label, word],
                                                                       (len(labelled_words) - label_word_freq[label, word]))

    high_info_words = set([word for word, score in word_scores.items() if score > min_score])
    return high_info_words

# Get movie review categories
labels = movie_reviews.categories()

# Create labeled words for each category
labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]

# Get high information words
high_info_words = set(high_information_words(labeled_words))

# Define a feature detector using high information words
def bag_of_words_in_set(words, goodwords):
    return bag_of_words(words, goodwords)

# Define a feature extractor
def bag_of_words(words, wordlist):
    return dict([(word, True) for word in words if word in wordlist])

# Create labeled feature sets using the feature detector
lfeats = [(bag_of_words_in_set(words, high_info_words), label) for label, words in labeled_words]

# Split the labeled feature sets into training and testing sets
split_ratio = 0.8
split = int(len(lfeats) * split_ratio)
train_feats = lfeats[:split]
test_feats = lfeats[split:]

# Train a Naive Bayes classifier
nb_classifier = NaiveBayesClassifier.train(train_feats)

# Calculate accuracy of the Naive Bayes classifier
nb_accuracy = accuracy(nb_classifier, test_feats)
print("Naive Bayes Accuracy:", nb_accuracy)

# Convert the test_feats to the required format
test_feats_sk = [(feats, label) for feats, label in test_feats]

# Make predictions using the Naive Bayes classifier
nb_predictions = [nb_classifier.classify(feats) for feats, _ in test_feats_sk]

# Extract true labels and predicted labels
true_labels = [label for _, label in test_feats_sk]
predicted_labels = nb_predictions

# Calculate precision and recall for the Naive Bayes classifier
nb_precision_pos = precision_score(true_labels, predicted_labels, pos_label='pos')
nb_recall_pos = recall_score(true_labels, predicted_labels, pos_label='pos')

nb_precision_neg = precision_score(true_labels, predicted_labels, pos_label='neg')
nb_recall_neg = recall_score(true_labels, predicted_labels, pos_label='neg')

print("Naive Bayes Precision (pos):", nb_precision_pos)
print("Naive Bayes Recall (pos):", nb_recall_pos)
print("Naive Bayes Precision (neg):", nb_precision_neg)
print("Naive Bayes Recall (neg):", nb_recall_neg)
