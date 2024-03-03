# Import the treebank dataset from NLTK
from nltk.corpus import treebank

# Import the shallow_tree function from your custom module (transforms)
from transforms import shallow_tree

# Get a deep tree from the treebank dataset
deep_tree = treebank.parsed_sents()[0]

# Create a shallow tree by collapsing certain nodes
shallow_tree_result = shallow_tree(deep_tree)

# Print the shallow tree
print(shallow_tree_result)

# Check the height of the original deep tree
print(deep_tree.height())

# Check the height of the shallow tree
print(shallow_tree_result.height())
