from nltk.tag import SequentialBackoffTagger
from nltk.corpus import wordnet
from nltk.probability import FreqDist
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, ClassifierBasedPOSTagger
from nltk.classify import MaxentClassifier
from nltk.tag.sequential import ClassifierBasedTagger
from nltk.tag import DefaultTagger
from nltk.corpus import names
from nltk.corpus import treebank
from nltk.probability import FreqDist

train_sents = treebank.tagged_sents()[:3000]
test_sents = treebank.tagged_sents()[3000:]

# Define a WordNetTagger class that inherits from SequentialBackoffTagger
class WordNetTagger(SequentialBackoffTagger):
    '''
    >>> wt = WordNetTagger()
    >>> wt.tag(['food', 'is', 'great'])
    [('food', 'NN'), ('is', 'VB'), ('great', 'JJ')]
    '''

    def __init__(self, *args, **kwargs):
        SequentialBackoffTagger.__init__(self, *args, **kwargs)
        self.wordnet_tag_map = {'n': 'NN', 's': 'JJ', 'a': 'JJ', 'r': 'RB', 'v': 'VB'}

    def choose_tag(self, tokens, index, history):
        word = tokens[index]
        fd = FreqDist()
        for synset in wordnet.synsets(word):
            fd.update([synset.pos()])  # Use update instead of inc
        
        if len(fd) > 0:
            return self.wordnet_tag_map.get(fd.max())
        else:
            return None

# Create a WordNetTagger instance
wn_tagger = WordNetTagger()

# Evaluate the WordNetTagger
wn_tagger.evaluate(train_sents)

# Import required modules
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.classify import MaxentClassifier

# Create a MaxentClassifier-based POS Tagger
tagger = ClassifierBasedPOSTagger(train=train_sents, classifier_builder=MaxentClassifier.train)

# Evaluate the MaxentClassifier-based POS Tagger
tagger.evaluate(test_sents)

# Define a unigram feature detector
def unigram_feature_detector(tokens, index, history):
    return {'word': tokens[index]}

# Create a ClassifierBasedTagger with the unigram feature detector
tagger = ClassifierBasedTagger(train=train_sents, feature_detector=unigram_feature_detector)

# Evaluate the ClassifierBasedTagger with unigram features
tagger.evaluate(test_sents)

# Create a DefaultTagger for handling unknown words
default = DefaultTagger('NN')

# Create a ClassifierBasedPOSTagger with a cutoff probability
tagger = ClassifierBasedPOSTagger(train=train_sents, backoff=default, cutoff_prob=0.3)

# Evaluate the ClassifierBasedPOSTagger with a cutoff probability
tagger.evaluate(test_sents)
