# Import the necessary modules and classes
import nltk
nltk.download('conll2000')
from nltk.chunk import RegexpParser

# Define a chunker using regular expressions
chunker = RegexpParser(r'''
    NP:
    {<DT>?<NN.*>+}   # chunk optional determiner with nouns
    <JJ>{}<NN.*>     # merge adjective with noun chunk
    PP:
    {<IN>}           # chunk preposition
    VP:
    {<MD>?<VB.*>}    # chunk optional modal with verb
''')

# Import the conll2000 dataset and evaluate the chunker
from nltk.corpus import conll2000

# Evaluate the chunker on conll2000 dataset and calculate accuracy
score = chunker.evaluate(conll2000.chunked_sents())
print(score.accuracy())  # Print the accuracy score

# Import the treebank_chunk dataset and evaluate the chunker
from nltk.corpus import treebank_chunk

# Evaluate the chunker on treebank_chunk dataset and calculate accuracy
treebank_score = chunker.evaluate(treebank_chunk.chunked_sents())
print(treebank_score.accuracy())  # Print the accuracy score

# Calculate precision and recall
print(score.precision())
print(score.recall())

# Get the number of missed, incorrect, correct, and guessed chunks
print(len(score.missed()))
print(len(score.incorrect()))
print(len(score.correct()))
print(len(score.guessed()))
