# Import the filter_insignificant function from your custom module (transforms)
from transforms import filter_insignificant

# Define a list of tagged words
tagged_words = [('your', 'PRP$'), ('book', 'NN'), ('is', 'VBZ'), ('great', 'JJ')]

# Use filter_insignificant to remove words with specified tag suffixes
filtered_words = filter_insignificant(tagged_words, tag_suffixes=['PRP', 'PRP$'])

# Print the filtered words
print(filtered_words)
