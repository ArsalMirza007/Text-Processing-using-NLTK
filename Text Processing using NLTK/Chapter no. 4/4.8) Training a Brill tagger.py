from nltk.tag import tnt
from nltk.tag import DefaultTagger
from nltk.corpus import treebank

# Load the Treebank corpus
train_sents = treebank.tagged_sents()[:3000]
test_sents = treebank.tagged_sents()[3000:]

# Train a TnT tagger
tnt_tagger = tnt.TnT()
tnt_tagger.train(train_sents)

# Evaluate the TnT tagger
tnt_accuracy = tnt_tagger.evaluate(test_sents)
print(f'TnT Tagger Accuracy (Default): {tnt_accuracy:.2%}')

# Create a DefaultTagger for handling unknown words
unk = DefaultTagger('NN')

# Train a TnT tagger with the DefaultTagger as a fallback for unknown words
tnt_tagger = tnt.TnT(unk=unk, Trained=True)
tnt_tagger.train(train_sents)

# Evaluate the TnT tagger with the DefaultTagger
tnt_accuracy_with_unk = tnt_tagger.evaluate(test_sents)
print(f'TnT Tagger Accuracy (With DefaultTagger for Unknown): {tnt_accuracy_with_unk:.2%}')

# Train a TnT tagger with a specified vocabulary size (N)
tnt_tagger = tnt.TnT(N=100)
tnt_tagger.train(train_sents)

# Evaluate the TnT tagger with a limited vocabulary size
tnt_accuracy_with_limited_vocab = tnt_tagger.evaluate(test_sents)
print(f'TnT Tagger Accuracy (With Limited Vocabulary): {tnt_accuracy_with_limited_vocab:.2%}')
