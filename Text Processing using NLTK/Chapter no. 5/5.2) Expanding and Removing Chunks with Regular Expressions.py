# Import necessary modules and classes
from nltk.chunk.regexp import ChunkRule, ExpandLeftRule, ExpandRightRule, UnChunkRule
from nltk.chunk import RegexpChunkParser

# Define ChunkRules for chunking and unchunking
ur = ChunkRule('<NN>', 'single noun')               # Chunk a single noun
el = ExpandLeftRule('<DT>', '<NN>', 'get left determiner')  # Expand left to include determiner
er = ExpandRightRule('<NN>', '<NNS>', 'get right plural noun')  # Expand right to include plural noun
un = UnChunkRule('<DT><NN.*>*', 'unchunk everything')   # Unchunk everything matching the pattern

# Create a RegexpChunkParser with the defined rules
chunker = RegexpChunkParser([ur, el, er, un])

# Define a sentence
sent = [('the', 'DT'), ('sushi', 'NN'), ('rolls', 'NNS')]

# Parse the sentence using the chunker
result = chunker.parse(sent)
print(result)  # Print the parsed result

# Import necessary modules and classes for ChunkString
from nltk.chunk.regexp import ChunkString
from nltk.tree import Tree

# Create a ChunkString from the sentence
cs = ChunkString(Tree('S', sent))
print(cs)  # Print the initial ChunkString

# Apply ChunkRule to chunk single nouns
ur.apply(cs)
print(cs)  # Print the ChunkString after applying the ChunkRule

# Apply ExpandLeftRule to include left determiners
el.apply(cs)
print(cs)  # Print the ChunkString after applying ExpandLeftRule

# Apply ExpandRightRule to include right plural nouns
er.apply(cs)
print(cs)  # Print the ChunkString after applying ExpandRightRule

# Apply UnChunkRule to unchunk everything matching the pattern
un.apply(cs)
print(cs)  # Print the ChunkString after applying UnChunkRule
