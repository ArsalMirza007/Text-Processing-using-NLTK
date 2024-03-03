import nltk
nltk.download('webtext')
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Convert words to lowercase
words = [w.lower() for w in webtext.words('grail.txt')]

# Create a BigramCollocationFinder from the words
bcf = BigramCollocationFinder.from_words(words)

# Find the top 4 bigram collocations using likelihood ratio
top_bigrams = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
print(top_bigrams)
# Output: [("'", 's'), ('arthur', ':'), ('#', '1'), ("'", 't')]

from nltk.corpus import stopwords

# Create a set of English stopwords
stopset = set(stopwords.words('english'))

# Define a filter to remove stopwords and short words
filter_stops = lambda w: len(w) < 3 or w in stopset

# Apply the word filter to the BigramCollocationFinder
bcf.apply_word_filter(filter_stops)

# Find the top 4 bigram collocations after applying the filter
top_filtered_bigrams = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)
print(top_filtered_bigrams)
# Output: [('black', 'knight'), ('clop', 'clop'), ('head', 'knight'), ('mumble', 'mumble')]

from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

# Convert words to lowercase for trigram analysis
words = [w.lower() for w in webtext.words('singles.txt')]

# Create a TrigramCollocationFinder from the words
tcf = TrigramCollocationFinder.from_words(words)

# Apply the word filter to remove stopwords and short words
tcf.apply_word_filter(filter_stops)

# Apply a frequency filter to keep trigrams that appear at least 3 times
tcf.apply_freq_filter(3)

# Find the top 4 trigram collocations using likelihood ratio
top_trigrams = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 4)
print(top_trigrams)
# Output: [('long', 'term', 'relationship')]
