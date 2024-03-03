from nltk.tag import DefaultTagger, UnigramTagger
import pickle
from nltk.corpus import treebank

# Load the training and testing data from the treebank corpus
train_sents = treebank.tagged_sents()[:3000]
test_sents = treebank.tagged_sents()[3000:]

# Create a DefaultTagger for nouns (NN)
tagger1 = DefaultTagger('NN')

# Train an UnigramTagger with a backoff to the DefaultTagger
tagger2 = UnigramTagger(train_sents, backoff=tagger1)

# Evaluate the UnigramTagger on test data
accuracy = tagger2.evaluate(test_sents)
print("Tagger Accuracy:", accuracy)

# Check if the DefaultTagger is included in the backoff chain of the UnigramTagger
print("Is DefaultTagger included in tagger1's backoff chain?", tagger1._taggers == [tagger1])

# Check if the DefaultTagger is included in the backoff chain of the UnigramTagger
print("Is DefaultTagger included in tagger2's backoff chain?", tagger2._taggers == [tagger2, tagger1])

# Serialize the tagger using pickle and save it to a file
with open('tagger.pickle', 'wb') as f:
    pickle.dump(tagger2, f)

# Deserialize the tagger from the saved file
with open('tagger.pickle', 'rb') as f:
    loaded_tagger = pickle.load(f)

# Now 'loaded_tagger' is the same as 'tagger2'
