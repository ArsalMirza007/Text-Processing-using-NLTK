from nltk.corpus import wordnet

# Get the first synset for the word 'cookbook'
syn = wordnet.synsets('cookbook')[0]

# Print the name of the synset
print(syn.name())
# Output: 'cookbook.n.01'

# Print the definition of the synset
print(syn.definition())
# Output: 'a book of recipes and cooking directions'

# Get the synset for 'cookbook' using its full name
print(wordnet.synset('cookbook.n.01'))
# Output: Synset('cookbook.n.01')

# Print examples for the first synset of 'cooking'
print(wordnet.synsets('cooking')[0].examples())
# Output: ['cooking can be a great art', 'people are needed who have experience in cookery', 'he left the preparation of meals to his wife']

# Get the hypernyms of the 'cookbook' synset
print(syn.hypernyms())
# Output: [Synset('reference_book.n.01')]

# Get the hyponyms of the first hypernym of 'cookbook'
print(syn.hypernyms()[0].hyponyms())
# Output: [Synset('encyclopedia.n.01'), Synset('directory.n.01'), Synset('source_book.n.01'), Synset('handbook.n.01'), Synset('instruction_book.n.01'), Synset('cookbook.n.01'), Synset('annual.n.02'), Synset('atlas.n.02'), Synset('wordbook.n.01')]

# Get the root hypernyms of 'cookbook'
print(syn.root_hypernyms())

# Get hypernym paths for 'cookbook'
print(syn.hypernym_paths())

# Get the number of synsets for 'great'
print(len(wordnet.synsets('great')))

# Get the number of noun synsets for 'great'
print(len(wordnet.synsets('great', pos='n')))

# Get the number of adjective synsets for 'great'
print(len(wordnet.synsets('great', pos='a')))
