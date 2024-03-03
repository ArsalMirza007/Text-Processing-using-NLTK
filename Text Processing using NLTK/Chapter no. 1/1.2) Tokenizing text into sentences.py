from nltk.tokenize import sent_tokenize
para = "Hello World. It's good to see you. Thanks for buying this book."

# Using sent_tokenize
sentences = sent_tokenize(para)
print(sentences)  # This will display the output in the IDLE shell.

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Using tokenizer.tokenize
sentences_with_tokenizer = tokenizer.tokenize(para)
print(sentences_with_tokenizer)  # This will display the output in the IDLE shell.
