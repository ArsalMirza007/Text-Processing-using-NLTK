from nltk.corpus import wordnet

# Get the first synset for the word 'cookbook'
syn = wordnet.synsets('cookbook')[0]

# Get the lemmas of the synset
lemmas = syn.lemmas()

# Print the number of lemmas
print(len(lemmas))  # Output: 2

# Print the name of the first lemma
print(lemmas[0].name())  # Output: 'cookbook'

# Print the name of the second lemma
print(lemmas[1].name())  # Output: 'cookery_book'

# Check if the synset of the first lemma is the same as the synset of the second lemma
print(lemmas[0].synset() == lemmas[1].synset())  # Output: True

# Print names of all lemmas in the synset
print([lemma.name() for lemma in syn.lemmas()])  # Output: ['cookbook', 'cookery_book']

# Initialize an empty list for synonyms
synonyms = []

# Iterate through synsets of 'book' and add lemma names to the synonyms list
for syn in wordnet.synsets('book'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())

# Print the number of synonyms (including duplicates)
print(len(synonyms))  # Output: 38

# Print the number of unique synonyms (removing duplicates)
print(len(set(synonyms)))  # Output: 25

# Get the synset for 'good.n.02'
gn2 = wordnet.synset('good.n.02')

# Print the definition of 'good.n.02'
print(gn2.definition())  # Output: 'moral excellence or admirableness'

# Get the antonym of 'good.n.02' (which is 'evil')
evil = gn2.lemmas()[0].antonyms()[0]

# Print the name of the antonym
print(evil.name())  # Output: 'evil'

# Print the definition of the antonym ('evil')
print(evil.synset().definition())  # Output: 'the quality of being morally wrong in principle or practice'

# Get the synset for 'good.a.01'
ga1 = wordnet.synset('good.a.01')

# Print the definition of 'good.a.01'
print(ga1.definition())  # Output: 'having desirable or positive qualities especially those suitable for a thing specified'

# Get the antonym of 'good.a.01' (which is 'bad')
bad = ga1.lemmas()[0].antonyms()[0]

# Print the name of the antonym
print(bad.name())  # Output: 'bad'

# Print the definition of the antonym ('bad')
print(bad.synset().definition())  # Output: 'having undesirable or negative qualities'
