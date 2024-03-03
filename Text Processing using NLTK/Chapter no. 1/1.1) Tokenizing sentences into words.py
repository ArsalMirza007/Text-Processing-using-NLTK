from nltk.tokenize import word_tokenize

# Tokenize using word_tokenize
tokens_word_tokenize = word_tokenize('Hello World.')
print(tokens_word_tokenize)  # Print the tokenized result
# Output: ['Hello', 'World', '.']

from nltk.tokenize import TreebankWordTokenizer

# Create a TreebankWordTokenizer instance
tokenizer = TreebankWordTokenizer()

# Tokenize using TreebankWordTokenizer
tokens_treebank = tokenizer.tokenize('Hello World.')
print(tokens_treebank)  # Print the tokenized result
# Output: ['Hello', 'World', '.']

# Tokenize using word_tokenize with a contraction
tokens_word_tokenize_contr = word_tokenize("can't")
print(tokens_word_tokenize_contr)  # Print the tokenized result
# Output: ['ca', "n't"]

from nltk.tokenize import word_tokenize

# Tokenize using word_tokenize with a contraction
text = "Can't is a contraction."
tokens_word_tokenize = word_tokenize(text)
print(tokens_word_tokenize)  # Print the tokenized result
# Output: ['Ca', "n't", 'is', 'a', 'contraction', '.']

