from nltk.stem import WordNetLemmatizer

# Create a WordNetLemmatizer instance
lemmatizer = WordNetLemmatizer()

# Lemmatize the word 'cooking' without specifying the part of speech
lemma1 = lemmatizer.lemmatize('cooking')
print(lemma1)  # Output: 'cooking'

# Lemmatize the word 'cooking' with specified part of speech as 'v' (verb)
lemma2 = lemmatizer.lemmatize('cooking', pos='v')
print(lemma2)  # Output: 'cook'

# Lemmatize the word 'cookbooks' without specifying the part of speech
lemma3 = lemmatizer.lemmatize('cookbooks')
print(lemma3)  # Output: 'cookbook'

from nltk.stem import PorterStemmer

# Create a PorterStemmer instance
stemmer = PorterStemmer()

# Stem the word 'believes'
stem1 = stemmer.stem('believes')
print(stem1)  # Output: 'believ'

# Lemmatize the word 'believes'
lemma4 = lemmatizer.lemmatize('believes')
print(lemma4)  # Output: 'belief'

# Stem the word 'buses'
stem2 = stemmer.stem('buses')
print(stem2)  # Output: 'buse'

# Lemmatize the word 'buses'
lemma5 = lemmatizer.lemmatize('buses')
print(lemma5)  # Output: 'bus'

# Stem the word 'bus'
stem3 = stemmer.stem('bus')
print(stem3)  # Output: 'bu'
