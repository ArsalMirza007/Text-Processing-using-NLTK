from nltk.tag import UnigramTagger
from nltk.corpus import treebank

# Load the training sentences
train_sents = treebank.tagged_sents()[:3000]

# Initialize the UnigramTagger with training data
tagger = UnigramTagger(train_sents)

# Example: Tagging a sentence
sentence = treebank.sents()[0]
tagged_sentence = tagger.tag(sentence)
print(tagged_sentence)

# Evaluating the tagger on test data
test_sents = treebank.tagged_sents()[3000:]
accuracy = tagger.evaluate(test_sents)
print("Tagger Accuracy:", accuracy)

# Customizing the tagger by providing a model
custom_tagger = UnigramTagger(model={'Pierre': 'NN'})
tagged_custom_sentence = custom_tagger.tag(sentence)
print(tagged_custom_sentence)

# Customizing the tagger with a cutoff frequency
tagger_with_cutoff = UnigramTagger(train_sents, cutoff=3)
accuracy_with_cutoff = tagger_with_cutoff.accuracy(test_sents)  # Use accuracy() function
print("Tagger Accuracy with Cutoff:", accuracy_with_cutoff)
