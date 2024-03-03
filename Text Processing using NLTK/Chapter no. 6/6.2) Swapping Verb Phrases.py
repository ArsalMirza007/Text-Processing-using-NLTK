# Import the swap_verb_phrase and filter_insignificant functions from your custom module (transforms)
from transforms import swap_verb_phrase, filter_insignificant

# Define a list of tagged words
tagged_words = [('this', 'DT'), ('gripping', 'VBG'), ('book', 'NN'), ('is', 'VBZ'), ('fantastic', 'JJ')]

# Swap the verb phrase and filter insignificant words
swapped_filtered_words = swap_verb_phrase(filter_insignificant(tagged_words))

# Print the swapped and filtered words
print(swapped_filtered_words)

# Swap the verb phrase in the original list and then filter insignificant words
filtered_swapped_words = filter_insignificant(swap_verb_phrase(tagged_words))

# Print the filtered and swapped words
print(filtered_swapped_words)
