# Import the necessary modules and classes
from nltk.corpus import treebank_chunk

# Get a chunked tree from the treebank_chunk dataset
tree = treebank_chunk.chunked_sents()[0]

# Extract and join the leaves (words) from the tree to form a sentence
sentence = ' '.join([w for w, t in tree.leaves()])

# Print the sentence
print(sentence)

# Import the chunk_tree_to_sent function from your custom module (transforms)
from transforms import chunk_tree_to_sent

# Convert the chunked tree to a sentence using the chunk_tree_to_sent function
converted_sentence = chunk_tree_to_sent(tree)

# Print the converted sentence
print(converted_sentence)
