from nltk.corpus import treebank
from nltk.tag import BigramTagger, TrigramTagger
from nltk.tag import DefaultTagger, UnigramTagger
from nltk.tag import NgramTagger

# Load the training and testing data from the treebank corpus
train_sents = treebank.tagged_sents()[:3000]
test_sents = treebank.tagged_sents()[3000:]

# Train a BigramTagger and evaluate its performance
bitagger = BigramTagger(train_sents)
bitagger_accuracy = bitagger.evaluate(test_sents)
print("BigramTagger Accuracy:", bitagger_accuracy)

# Train a TrigramTagger and evaluate its performance
tritagger = TrigramTagger(train_sents)
tritagger_accuracy = tritagger.evaluate(test_sents)
print("TrigramTagger Accuracy:", tritagger_accuracy)

# Define a backoff_tagger function
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff

# Train taggers using backoff with DefaultTagger
backoff = DefaultTagger('NN')

# Create a tagger using backoff_tagger with UnigramTagger, BigramTagger, and TrigramTagger
tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=backoff)

# Evaluate the combined tagger
tagger_accuracy = tagger.evaluate(test_sents)
print("Combined Tagger Accuracy:", tagger_accuracy)

# Check if the last tagger in the chain is the same as the specified backoff
print("Is DefaultTagger included in the backoff chain of the combined tagger?", tagger._taggers[-1] == backoff)

# Check the taggers in the chain
print("Is the first tagger a TrigramTagger?", isinstance(tagger._taggers[0], TrigramTagger))
print("Is the second tagger a BigramTagger?", isinstance(tagger._taggers[1], BigramTagger))

# Train a QuadgramTagger and evaluate its performance
quadtagger = NgramTagger(4, train_sents)
quadtagger_accuracy = quadtagger.evaluate(test_sents)
print("QuadgramTagger Accuracy:", quadtagger_accuracy)

# Define a custom QuadgramTagger class
class QuadgramTagger(NgramTagger):
    def __init__(self, *args, **kwargs):
        NgramTagger.__init__(self, 4, *args, **kwargs)

# Train a tagger using backoff_tagger with UnigramTagger, BigramTagger, TrigramTagger, and QuadgramTagger
quadtagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger, QuadgramTagger], backoff=backoff)

# Evaluate the combined tagger with QuadgramTagger
quadtagger_accuracy = quadtagger.evaluate(test_sents)
print("Combined Tagger with QuadgramTagger Accuracy:", quadtagger_accuracy)
