from nltk.corpus import reuters
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics import precision, recall, f_measure

# Get the number of Reuters categories
num_categories = len(reuters.categories())
print(num_categories)

# Create a function to find high information words
def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    word_scores = {}
    label_word_freq = FreqDist()
    total_word_freq = FreqDist()

    for label, words in labelled_words:
        label_word_freq[label] += 1
        total_word_freq.update(words)

    for label, words in labelled_words:
        for word in words:
            if len(word) > 2:
                n_ii = label_word_freq[label].get(word, 0)
                n_ix = total_word_freq.freq(word)
                n_xi = label_word_freq[label].N()
                n_xx = total_word_freq.N()
                
                score = score_fn(n_ii, (n_ix, n_xi), n_xx)
                word_scores[word] = word_scores.get(word, 0) + score

    high_info_words = set([word for word, score in word_scores.items() if score > min_score])
    return high_info_words

# Get high information words for Reuters dataset
labeled_words = [(l, reuters.words(categories=[l])) for l in reuters.categories()]
rwords = high_information_words(labeled_words)

# Define a feature detector using high information words
def bag_of_words_in_set(words, wordlist):
    return bag_of_words(words, wordlist)

def bag_of_words(words, wordlist):
    return dict([(word, True) for word in words if word in wordlist])

# Create training and testing feature sets for Reuters dataset
multi_train_feats = []
multi_test_feats = []

for label, words in labeled_words:
    features = bag_of_words_in_set(words, rwords)
    split_index = int(len(features) * 0.8)
    train_feats = [(feat, label) for feat in features.items()[:split_index]]
    test_feats = [(feat, label) for feat in features.items()[split_index:]]
    multi_train_feats.extend(train_feats)
    multi_test_feats.extend(test_feats)

# Train binary classifiers for each category
classifiers = {}
for category in reuters.categories():
    train_feats = [(feat, label) for feat, label in multi_train_feats if label == category]
    classifiers[category] = SklearnClassifier(LogisticRegression()).train(train_feats)

# Calculate precision, recall, and average mutual information discrepancy
predictions = {}
true_labels = [label for (_, label) in multi_test_feats]
avg_md = 0.0

for category in classifiers:
    classifier = classifiers[category]
    test_feats = [(feat, label) for feat, label in multi_test_feats if label == category]
    predictions[category] = [classifier.classify(feat) for feat, _ in test_feats]
    true_labels_category = [label for (_, label) in test_feats]
    
    precision_category = precision(true_labels_category, predictions[category], category)
    recall_category = recall(true_labels_category, predictions[category], category)
    
    avg_md += 1 - f_measure(true_labels_category, predictions[category], category)
    
    print(f"Precision for '{category}': {precision_category}")
    print(f"Recall for '{category}': {recall_category}")

avg_md /= num_categories
print(f"Average Mutual Information Discrepancy: {avg_md}")
