# Import the necessary modules and classes
from chunkers import ClassifierChunker
from nltk.corpus import treebank_chunk, conll2000
from nltk.classify import MaxentClassifier

# Define train and test datasets for treebank_chunk
train_chunks_treebank = treebank_chunk.chunked_sents()[:3000]
test_chunks_treebank = treebank_chunk.chunked_sents()[3000:]

# Define train and test datasets for conll2000
conll_train = conll2000.chunked_sents('train.txt')
conll_test = conll2000.chunked_sents('test.txt')

# Create a ClassifierChunker and train it on treebank_chunk train data
chunker_treebank = ClassifierChunker(train_chunks_treebank)

# Evaluate the chunker on treebank_chunk test data and calculate accuracy, precision, and recall
score_treebank = chunker_treebank.evaluate(test_chunks_treebank)
print("Treebank Chunker:")
print("Accuracy:", score_treebank.accuracy())
print("Precision:", score_treebank.precision())
print("Recall:", score_treebank.recall())

# Create a ClassifierChunker and train it on conll2000 train data
chunker_conll = ClassifierChunker(conll_train)

# Evaluate the chunker on conll2000 test data and calculate accuracy, precision, and recall
score_conll = chunker_conll.evaluate(conll_test)
print("\nCONLL Chunker:")
print("Accuracy:", score_conll.accuracy())
print("Precision:", score_conll.precision())
print("Recall:", score_conll.recall())

# Define a builder function for MaxentClassifier
builder = lambda toks: MaxentClassifier.train(toks, trace=0, max_iter=10, min_lldelta=0.01)

# Create a ClassifierChunker with MaxentClassifier and train it on treebank_chunk train data
me_chunker = ClassifierChunker(train_chunks_treebank, classifier_builder=builder)

# Evaluate the chunker on treebank_chunk test data and calculate accuracy, precision, and recall
score_me = me_chunker.evaluate(test_chunks_treebank)
print("\nMaxent Classifier Chunker:")
print("Accuracy:", score_me.accuracy())
print("Precision:", score_me.precision())
print("Recall:", score_me.recall())
