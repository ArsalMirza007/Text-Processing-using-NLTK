# Import the necessary modules and classes
from chunkers import ieer_chunked_sents, ClassifierChunker
from nltk.corpus import treebank_chunk
from nltk.corpus import ieer  # Correct import statement

# Get the IEER chunks and calculate their length
ieer_chunks = list(ieer_chunked_sents())
print(len(ieer_chunks))

# Create a ClassifierChunker and train it on a subset of IEER chunks
chunker = ClassifierChunker(ieer_chunks[:80])

# Parse a sentence from the treebank and chunk it
parsed_sentence = chunker.parse(treebank_chunk.tagged_sents()[0])
print(parsed_sentence)

# Evaluate the chunker on a subset of IEER chunks and calculate accuracy, precision, and recall
score = chunker.evaluate(ieer_chunks[80:])
print(score.accuracy())
print(score.precision())
print(score.recall())

# Access the headline of the first IEER parsed document
print(ieer.parsed_docs()[0].headline)
