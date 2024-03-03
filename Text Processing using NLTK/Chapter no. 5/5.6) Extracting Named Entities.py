# Import the necessary modules and classes
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.chunk import ne_chunk, ne_chunk_sents  # Import ne_chunk_sents
from nltk.corpus import treebank_chunk  # Import the treebank_chunk dataset

# Perform named entity recognition on a tagged sentence
result = ne_chunk(treebank_chunk.tagged_sents()[0])

# Print the result
print(result)

# Import the sub_leaves function from your custom module (chunkers)
from chunkers import sub_leaves

# Extract leaves of specified entity types (e.g., PERSON)
person_leaves = sub_leaves(result, 'PERSON')
print(person_leaves)

# Extract leaves of specified entity types (e.g., ORGANIZATION)
organization_leaves = sub_leaves(result, 'ORGANIZATION')
print(organization_leaves)

# Perform named entity recognition on multiple tagged sentences
trees = ne_chunk_sents(treebank_chunk.tagged_sents()[:10])

# Extract leaves of specified entity types (e.g., ORGANIZATION) for each sentence
organization_leaves_list = [sub_leaves(t, 'ORGANIZATION') for t in trees]
print(organization_leaves_list)

# Perform binary named entity recognition on a tagged sentence
binary_result = ne_chunk(treebank_chunk.tagged_sents()[0], binary=True)
print(binary_result)

# Extract leaves of specified entity types (e.g., NE) for binary recognition
binary_leaves = sub_leaves(binary_result, 'NE')
print(binary_leaves)
