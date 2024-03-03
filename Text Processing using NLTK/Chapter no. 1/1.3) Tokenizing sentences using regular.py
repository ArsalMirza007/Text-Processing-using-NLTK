from nltk.tokenize import RegexpTokenizer

# Create a RegexpTokenizer instance for words and contractions
tokenizer = RegexpTokenizer("[\w']+")

# Tokenize using the RegexpTokenizer
tokens_word_contr = tokenizer.tokenize("Can't is a contraction.")
print(tokens_word_contr)  # Print the tokenized result
# Output: ["Can't", 'is', 'a', 'contraction']

from nltk.tokenize import regexp_tokenize

# Tokenize using regexp_tokenize for words and contractions
tokens_regexp = regexp_tokenize("Can't is a contraction.", "[\w']+")
print(tokens_regexp)  # Print the tokenized result
# Output: ["Can't", 'is', 'a', 'contraction']

# Create a RegexpTokenizer instance for whitespace separation with gaps=True
tokenizer = RegexpTokenizer('\s+', gaps=True)

# Tokenize using the RegexpTokenizer with whitespace separation
tokens_whitespace = tokenizer.tokenize("Can't is a contraction.")
print(tokens_whitespace)  # Print the tokenized result
# Output: ["Can't", 'is', 'a', 'contraction.']
