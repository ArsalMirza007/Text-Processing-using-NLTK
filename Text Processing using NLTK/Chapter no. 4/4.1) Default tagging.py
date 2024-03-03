import nltk
nltk.download('treebank')

from nltk.tag import DefaultTagger

# Create a DefaultTagger that tags everything as 'NN'
tagger = DefaultTagger('NN')

# Tag a list of words
tagged_words = tagger.tag(['Hello', 'World'])
print(tagged_words)

from nltk.corpus import treebank

# Load the Treebank dataset
test_sents = treebank.tagged_sents()[3000:]

# Evaluate the tagger on the test sentences
accuracy = tagger.evaluate(test_sents)
print(accuracy)

from nltk import pos_tag

# Define a list of sentences
sentences = [['Hello', 'world', '.'], ['How', 'are', 'you', '?']]

# Tag the sentences using pos_tag
batch_tags = [pos_tag(sentence) for sentence in sentences]

# Print the tagged sentences
for tags in batch_tags:
    print(tags)

