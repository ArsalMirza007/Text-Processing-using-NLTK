# Import the necessary corpus from NLTK
from nltk.corpus import treebank

# Import the convert_tree_labels function from your custom module (transforms)
from transforms import convert_tree_labels

# Define the mapping for tree label conversion
mapping = {'NP-SBJ': 'NP', 'NP-TMP': 'NP'}

# Apply the label conversion to a specific tree (in this case, the first parsed sentence from the treebank corpus)
converted_tree = convert_tree_labels(treebank.parsed_sents()[0], mapping)

# Print the resulting tree with updated labels
print(converted_tree)
