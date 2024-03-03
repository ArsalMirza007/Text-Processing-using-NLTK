# Import the necessary modules and classes
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tree import Tree

# Create a tree
t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])

# Convert the tree to CoNLL tags format
conll_tags = tree2conlltags(t)
print(conll_tags)

# Convert CoNLL tags format back to a tree
reconstructed_tree = conlltags2tree(conll_tags)
print(reconstructed_tree)

# Import the TagChunker class from your custom module (chunkers)
from chunkers import TagChunker

# Import the treebank_chunk dataset and split it into train and test sets
from nltk.corpus import treebank_chunk

train_chunks = treebank_chunk.chunked_sents()[:3000]
test_chunks = treebank_chunk.chunked_sents()[3000:]

# Create a TagChunker and train it on the train_chunks
chunker = TagChunker(train_chunks)

# Evaluate the chunker on test_chunks and calculate accuracy
score = chunker.evaluate(test_chunks)
print(score.accuracy())
print(score.precision())
print(score.recall())

# Import the conll2000 dataset and split it into train and test sets
from nltk.corpus import conll2000

conll_train = conll2000.chunked_sents('train.txt')
conll_test = conll2000.chunked_sents('test.txt')

# Create a TagChunker and train it on the conll_train dataset
chunker = TagChunker(conll_train)

# Evaluate the chunker on conll_test and calculate accuracy
score = chunker.evaluate(conll_test)
print(score.accuracy())
print(score.precision())
print(score.recall())

# Import the UnigramTagger from NLTK
from nltk.tag import UnigramTagger

# Create a TagChunker using UnigramTagger and evaluate it on test_chunks
uni_chunker = TagChunker(train_chunks, tagger_classes=[UnigramTagger])
score = uni_chunker.evaluate(test_chunks)
print(score.accuracy())
