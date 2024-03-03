from nltk.stem import PorterStemmer

# Create a PorterStemmer instance
stemmer = PorterStemmer()

# Stem the word 'cooking'
stem1 = stemmer.stem('cooking')
print(stem1)  # Output: 'cook'

# Stem the word 'cookery'
stem2 = stemmer.stem('cookery')
print(stem2)  # Output: 'cookeri'

from nltk.stem import LancasterStemmer

# Create a LancasterStemmer instance
stemmer = LancasterStemmer()

# Stem the word 'cooking'
stem3 = stemmer.stem('cooking')
print(stem3)  # Output: 'cook'

# Stem the word 'cookery'
stem4 = stemmer.stem('cookery')
print(stem4)  # Output: 'cookery'

from nltk.stem import RegexpStemmer

# Create a RegexpStemmer that removes 'ing' suffix
stemmer = RegexpStemmer('ing')

# Stem the word 'cooking'
stem5 = stemmer.stem('cooking')
print(stem5)  # Output: 'cook'

# Stem the word 'cookery'
stem6 = stemmer.stem('cookery')
print(stem6)  # Output: 'cookery'

# Stem the word 'ingleside'
stem7 = stemmer.stem('ingleside')
print(stem7)  # Output: 'leside'

from nltk.stem import SnowballStemmer

# List of supported languages by SnowballStemmer
supported_languages = SnowballStemmer.languages
print(supported_languages)
# Output: ('danish', 'dutch', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish')

# Create a SnowballStemmer for Spanish
spanish_stemmer = SnowballStemmer('spanish')

# Stem the word 'hola'
stem8 = spanish_stemmer.stem('hola')
print(stem8)  # Output: 'hol'
