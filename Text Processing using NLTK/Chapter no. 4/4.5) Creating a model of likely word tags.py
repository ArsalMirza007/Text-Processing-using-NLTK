from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger

# Define the backoff_tagger function
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff

# Define the word_tag_model function
def word_tag_model(words, tagged_words, limit=200):
    fd = FreqDist(words)
    most_freq = list(fd.keys())[:limit]  # Convert dict_keys to a list
    cfd = ConditionalFreqDist(tagged_words)
    return dict((word, cfd[word].max()) for word in most_freq)

# Load the training and testing data from the treebank corpus
train_sents = treebank.tagged_sents()[:3000]
test_sents = treebank.tagged_sents()[3000:]

# Create a word_tag model based on the treebank data
model = word_tag_model(treebank.words(), treebank.tagged_words())

# Create a tagger using the UnigramTagger with the word_tag model
tagger = UnigramTagger(model=model)

# Evaluate the UnigramTagger on the test data
print(tagger.evaluate(test_sents))

# Create a DefaultTagger with 'NN' as the default tag
default_tagger = DefaultTagger('NN')

# Create an UnigramTagger with the word_tag model and the DefaultTagger as a backoff
likely_tagger = UnigramTagger(model=model, backoff=default_tagger)

# Create a backoff tagger that combines UnigramTagger, BigramTagger, and TrigramTagger with likely_tagger as a backoff
tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=likely_tagger)

# Evaluate the combined tagger on the test data
print(tagger.evaluate(test_sents))

# Create a backoff tagger that combines UnigramTagger, BigramTagger, and TrigramTagger with DefaultTagger as a backoff
tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=default_tagger)

# Create an UnigramTagger with the word_tag model and the combined tagger as a backoff
likely_tagger = UnigramTagger(model=model, backoff=tagger)

# Evaluate the UnigramTagger with the combined tagger as a backoff on the test data
print(likely_tagger.evaluate(test_sents))
