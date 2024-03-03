import nltk
from nltk.tag import RegexpTagger

# Define your custom tagging patterns
patterns = [
    (r'^\d+$', 'CD'),       # Match digits as CD (cardinal numbers)
    (r'.*ing$', 'VBG'),     # Match words ending in 'ing' as VBG (gerunds)
    (r'.*ment$', 'NN'),     # Match words ending in 'ment' as NN (nouns)
    (r'.*ful$', 'JJ')       # Match words ending in 'ful' as JJ (adjectives)
]

# Load test sentences (you may need to replace this with your actual test data)
test_sents = nltk.corpus.treebank.tagged_sents()[3000:]

# Create a RegexpTagger using the defined patterns
tagger = RegexpTagger(patterns)

# Evaluate the tagger on the test sentences
accuracy = tagger.evaluate(test_sents)
print(f'Accuracy: {accuracy:.2%}')

