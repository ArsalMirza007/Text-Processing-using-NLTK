from nltk.corpus import stopwords

# Load the set of English stopwords
english_stops = set(stopwords.words('english'))

# Define a list of words
words = ["Can't", 'is', 'a', 'contraction']

# Use a list comprehension to filter out stopwords
filtered_words = [word for word in words if word not in english_stops]

print(filtered_words)  # Print the filtered words
# Output: ["Can't", 'contraction']

# Check the available stopwords lists
stopwords_list = stopwords.fileids()
print(stopwords_list)
# Output: ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', ...]
