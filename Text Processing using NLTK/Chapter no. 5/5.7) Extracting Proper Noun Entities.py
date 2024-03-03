# Import the necessary modules and classes
from nltk.chunk import RegexpParser
from chunkers import sub_leaves
from nltk.corpus import treebank_chunk

# Define the RegexpParser for extracting names
chunker = RegexpParser(r'''
    NAME:
        {<NNP>+}
''')

# Perform named entity recognition on a tagged sentence and extract 'NAME' entities
result = chunker.parse(treebank_chunk.tagged_sents()[0])
name_leaves = sub_leaves(result, 'NAME')
print(name_leaves)
