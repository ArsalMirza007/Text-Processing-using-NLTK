# Import necessary modules and functions
from nltk.corpus import movie_reviews
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
from nltk.probability import FreqDist, MLEProbDist, entropy

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

# Train a Decision Tree classifier
# Parameters:
# - binary=True: Indicates binary decision tree (two classes)
# - entropy_cutoff=0.8: Cutoff value for entropy-based splitting
# - depth_cutoff=5: Maximum depth for the decision tree
# - support_cutoff=30: Minimum number of samples to create a split
dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)

# Calculate the accuracy of the Decision Tree classifier on the test set
accuracy_score = accuracy(dt_classifier, test_feats)
print(accuracy_score)  # Output: 0.688

# Create a frequency distribution representing class labels ('pos' and 'neg')
fd = FreqDist({'pos': 30, 'neg': 10})

# Calculate the entropy of the class distribution using Maximum Likelihood Estimation (MLE)
entropy_score = entropy(MLEProbDist(fd))
print(entropy_score)  # Output: 0.8112781244591328

# Update the frequency distribution with a different 'neg' count
fd['neg'] = 25

# Recalculate the entropy with the updated frequency distribution
entropy_score = entropy(MLEProbDist(fd))
print(entropy_score)  # Output: 0.9940302114769565

# Update the frequency distribution to have equal 'pos' and 'neg' counts
fd['neg'] = 30

# Recalculate the entropy with the equal counts
entropy_score = entropy(MLEProbDist(fd))
print(entropy_score)  # Output: 1.0

# Update the frequency distribution with a minimal 'neg' count
fd['neg'] = 1

# Recalculate the entropy with the minimal 'neg' count
entropy_score = entropy(MLEProbDist(fd))
print(entropy_score)  # Output: 0.20559250818508304
