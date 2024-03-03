# Import the necessary modules and classes
import nltk
nltk.download('cess_esp')
from nltk.corpus import treebank
from transforms import flatten_deeptree

# Flatten a deep tree from the treebank dataset
flattened_tree = flatten_deeptree(treebank.parsed_sents()[0])

# Print the flattened tree
print(flattened_tree)

# Check the height of a shallow tree
print(flattened_tree.height())

# Import the Tree class from nltk.tree
from nltk.tree import Tree

# Check the height and POS tags of trees
print(Tree('NNP', ['Pierre']).height())
print(Tree('NNP', ['Pierre']).pos())
print(Tree('NP', [Tree('NNP', ['Pierre']), Tree('NNP', ['Vinken'])]).height())
print(Tree('NP', [Tree('NNP', ['Pierre']), Tree('NNP', ['Vinken'])]).pos())

# Import the tree2conlltags function from nltk.chunk.util
from nltk.chunk.util import tree2conlltags

# Convert the flattened tree to CoNLL tags
conll_tags = tree2conlltags(flattened_tree)

# Print the CoNLL tags
print(conll_tags)

# Import the cess_esp corpus and check the height of a deep tree and its flattened version
from nltk.corpus import cess_esp
print(cess_esp.parsed_sents()[0].height())
print(flatten_deeptree(cess_esp.parsed_sents()[0]).height())
