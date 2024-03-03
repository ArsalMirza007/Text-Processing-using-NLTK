from nltk.corpus import wordnet

# Get synsets for 'cookbook.n.01' and 'instruction_book.n.01'
cb = wordnet.synset('cookbook.n.01')
ib = wordnet.synset('instruction_book.n.01')

# Calculate Wu-Palmer Similarity between 'cookbook' and 'instruction_book'
similarity_cb_ib = cb.wup_similarity(ib)
print(similarity_cb_ib)  # Output: 0.9166666666666666

# Get the hypernym of 'cookbook'
ref = cb.hypernyms()[0]

# Calculate the shortest path distance from 'cookbook' to its hypernym
distance_cb_ref = cb.shortest_path_distance(ref)
print(distance_cb_ref)  # Output: 1

# Calculate the shortest path distance from 'instruction_book' to the same hypernym
distance_ib_ref = ib.shortest_path_distance(ref)
print(distance_ib_ref)  # Output: 1

# Calculate the shortest path distance between 'cookbook' and 'instruction_book'
distance_cb_ib = cb.shortest_path_distance(ib)
print(distance_cb_ib)  # Output: 2

# Get the first synset for 'dog'
dog = wordnet.synsets('dog')[0]

# Calculate Wu-Palmer Similarity between 'dog' and 'cookbook'
similarity_dog_cb = dog.wup_similarity(cb)
print(similarity_dog_cb)  # Output: 0.38095238095238093

# Find common hypernyms between 'cookbook' and 'dog'
common_hypernyms = cb.common_hypernyms(dog)
print(common_hypernyms)
# Output: [Synset('object.n.01'), Synset('whole.n.02'), Synset('physical_entity.n.01'), Synset('entity.n.01')]

# Get synsets for 'cook.v.01' and 'bake.v.02'
cook = wordnet.synset('cook.v.01')
bake = wordnet.synset('bake.v.02')

# Calculate Wu-Palmer Similarity between 'cook' and 'bake'
similarity_cook_bake = cook.wup_similarity(bake)
print(similarity_cook_bake)  # Output: 0.75

# Calculate path similarity between 'cookbook' and 'instruction_book'
path_similarity_cb_ib = cb.path_similarity(ib)
print(path_similarity_cb_ib)  # Output: 0.3333333333333333

# Calculate path similarity between 'cookbook' and 'dog'
path_similarity_cb_dog = cb.path_similarity(dog)
print(path_similarity_cb_dog)  # Output: 0.07142857142857142

# Calculate Leacock-Chodorow Similarity between 'cookbook' and 'instruction_book'
lch_similarity_cb_ib = cb.lch_similarity(ib)
print(lch_similarity_cb_ib)  # Output: 2.538973871058276

# Calculate Leacock-Chodorow Similarity between 'cookbook' and 'dog'
lch_similarity_cb_dog = cb.lch_similarity(dog)
print(lch_similarity_cb_dog)  # Output: 0.9985288301111273
