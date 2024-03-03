#alternative to the nltk.misc.babelfish module for text translation

from googletrans import Translator
from googletrans import LANGUAGES

# Create a Translator object
translator = Translator()

# Translate 'cookbook' from English to Spanish
translation1 = translator.translate('cookbook', src='en', dest='es')
print(translation1.text)  # Output: 'libro de cocina'

# Translate 'libro de cocina' from Spanish to English
translation2 = translator.translate('libro de cocina', src='es', dest='en')
print(translation2.text)  # Output: 'kitchen book'

# Translate 'cookbook' from English to German
translation3 = translator.translate('cookbook', src='en', dest='de')
print(translation3.text)  # Output: 'Kochbuch'

# Translate 'Kochbuch' from German to English
translation4 = translator.translate('Kochbuch', src='de', dest='en')
print(translation4.text)  # Output: 'cookbook'

# You can also use babelize by splitting the text into sentences and translating each sentence
text_to_translate = 'cookbook'
translations = []

for sentence in text_to_translate.split('. '):
    translation = translator.translate(sentence, src='en', dest='es')
    translations.append(translation.text)

for text in translations:
    print(text)

# Get a list of available languages
available_languages = LANGUAGES
print(list(available_languages.keys()))
