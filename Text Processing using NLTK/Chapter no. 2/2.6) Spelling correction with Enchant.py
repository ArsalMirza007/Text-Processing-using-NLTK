import enchant
from nltk.metrics import edit_distance

class SpellingReplacer(object):
    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        # Check if the word is already in the dictionary
        if self.spell_dict.check(word):
            return word
        # Get suggestions for the misspelled word
        suggestions = self.spell_dict.suggest(word)
        # If there are suggestions and the edit distance is within the threshold, replace the word
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

# Usage example
replacer = SpellingReplacer()
print(replacer.replace('cookbok'))  # Output: 'cookbook'

# Additional code snippets
d = enchant.Dict('en')
print(d.suggest('languege'))  # Example of using the enchant library for spell suggestions
print(enchant.list_languages())  # List available languages

# Using different dictionaries
dUS = enchant.Dict('en_US')
print(dUS.check('theater'))  # Check spelling in US English
dGB = enchant.Dict('en_GB')
print(dGB.check('theater'))  # Check spelling in UK English

# Custom Spelling Replacer
class CustomSpellingReplacer(SpellingReplacer):
    def __init__(self, spell_dict, max_dist=2):
        self.spell_dict = spell_dict
        self.max_dist = max_dist

# Example of using a custom dictionary with CustomSpellingReplacer
d = enchant.DictWithPWL('en_US', 'mywords.txt')
replacer = CustomSpellingReplacer(d)
print(replacer.replace('nltk'))
