from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC  # Import NuSVC here
from nltk.classify import accuracy

# Sample labeled feature sets (you should replace these with your actual data)
train_feats = [
    ({"feature1": True, "feature2": False}, "pos"),
    ({"feature1": False, "feature2": True}, "neg"),
    # Add more labeled feature sets here
]

test_feats = [
    ({"feature1": True, "feature2": False}, "pos"),
    ({"feature1": False, "feature2": True}, "neg"),
    # Add more labeled feature sets here
]

# Train a Multinomial Naive Bayes classifier
multinomial_nb_classifier = SklearnClassifier(MultinomialNB())
multinomial_nb_classifier.train(train_feats)

# Calculate the accuracy of the classifier on the test set
accuracy_score = accuracy(multinomial_nb_classifier, test_feats)
print("Multinomial Naive Bayes Accuracy:", accuracy_score)

# Train a Multinomial Naive Bayes classifier
multinomial_nb_classifier = SklearnClassifier(MultinomialNB())
multinomial_nb_classifier.train(train_feats)
accuracy_score = accuracy(multinomial_nb_classifier, test_feats)
print("Multinomial Naive Bayes Accuracy:", accuracy_score)

# Train a Bernoulli Naive Bayes classifier
bernoulli_nb_classifier = SklearnClassifier(BernoulliNB())
bernoulli_nb_classifier.train(train_feats)
accuracy_score = accuracy(bernoulli_nb_classifier, test_feats)
print("Bernoulli Naive Bayes Accuracy:", accuracy_score)

# Train a Logistic Regression classifier
logistic_regression_classifier = SklearnClassifier(LogisticRegression()).train(train_feats)
accuracy_score = accuracy(logistic_regression_classifier, test_feats)
print("Logistic Regression Accuracy:", accuracy_score)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SklearnClassifier(SVC()).train(train_feats)
accuracy_score = accuracy(svm_classifier, test_feats)
print("SVM Accuracy:", accuracy_score)

# Train a Linear Support Vector Machine (SVM) classifier
linear_svc_classifier = SklearnClassifier(LinearSVC()).train(train_feats)
accuracy_score = accuracy(linear_svc_classifier, test_feats)
print("Linear SVM Accuracy:", accuracy_score)

# Train a Nu-Support Vector Machine (SVM) classifier
nu_svc_classifier = SklearnClassifier(NuSVC()).train(train_feats)
accuracy_score = accuracy(nu_svc_classifier, test_feats)
print("Nu-SVM Accuracy:", accuracy_score)
