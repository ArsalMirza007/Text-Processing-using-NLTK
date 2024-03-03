from nltk.tag import AffixTagger
from nltk.corpus import treebank

# Obtain training sentences from the treebank corpus
train_sents = treebank.tagged_sents()[:3000]

# Train AffixTagger on training sentences
tagger = AffixTagger(train_sents)

# Evaluate AffixTagger on test sentences (replace with your actual test data)
test_sents = treebank.tagged_sents()[3000:]  # Modify this to your test data
accuracy = tagger.evaluate(test_sents)
print(f'Accuracy: {accuracy:.2%}')


# Create AffixTagger with a specific prefix length (3)
prefix_tagger = AffixTagger(train_sents, affix_length=3)

# Evaluate the prefix-based AffixTagger on test sentences
prefix_accuracy = prefix_tagger.evaluate(test_sents)
print(f'Accuracy (Prefix Length 3): {prefix_accuracy:.2%}')

# Create AffixTagger with a specific suffix length (-2)
suffix_tagger = AffixTagger(train_sents, affix_length=-2)

# Evaluate the suffix-based AffixTagger on test sentences
suffix_accuracy = suffix_tagger.evaluate(test_sents)
print(f'Accuracy (Suffix Length -2): {suffix_accuracy:.2%}')

# Create AffixTagger with a specific prefix length (3)
pre3_tagger = AffixTagger(train_sents, affix_length=3)

# Evaluate the prefix-based AffixTagger with backoff on test sentences
pre3_accuracy = pre3_tagger.evaluate(test_sents)
print(f'Accuracy (Prefix Length 3 with Backoff): {pre3_accuracy:.2%}')

# Create AffixTagger with a specific prefix length (2) and a backoff to pre3_tagger
pre2_tagger = AffixTagger(train_sents, affix_length=2, backoff=pre3_tagger)

# Evaluate the prefix-based AffixTagger with backoff on test sentences
pre2_accuracy = pre2_tagger.evaluate(test_sents)
print(f'Accuracy (Prefix Length 2 with Backoff): {pre2_accuracy:.2%}')

# Create AffixTagger with a specific suffix length (-2) and a backoff to pre2_tagger
suf2_tagger = AffixTagger(train_sents, affix_length=-2, backoff=pre2_tagger)

# Evaluate the suffix-based AffixTagger with backoff on test sentences
suf2_accuracy = suf2_tagger.evaluate(test_sents)
print(f'Accuracy (Suffix Length -2 with Backoff): {suf2_accuracy:.2%}')

# Create AffixTagger with a specific suffix length (-3) and a backoff to suf2_tagger
suf3_tagger = AffixTagger(train_sents, affix_length=-3, backoff=suf2_tagger)

# Evaluate the suffix-based AffixTagger with backoff on test sentences
suf3_accuracy = suf3_tagger.evaluate(test_sents)
print(f'Accuracy (Suffix Length -3 with Backoff): {suf3_accuracy:.2%}')
