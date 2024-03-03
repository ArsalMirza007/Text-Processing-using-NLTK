from nltk.corpus import wordnet

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

class AntonymReplacer(object):
    def replace(self, word, pos=None):
        antonyms = set()
        
        # Iterate through WordNet synsets to find antonyms
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())

        # If only one antonym is found, return it; otherwise, return None
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []

        while i < l:
            word = sent[i]

            # Check for the word 'not' and its antonym
            if word == 'not' and i + 1 < l:
                ant = self.replace(sent[i + 1])
                if ant:
                    words.append(ant)
                i += 2
                continue

            words.append(word)
            i += 1

        return words

# Example usage:
replacer = AntonymReplacer()

# Find antonyms for 'good'
print(replacer.replace('good'))

# Find antonyms for 'uglify'
print(replacer.replace('uglify'))

# Replace negations in a sentence
sent = ["let's", 'not', 'uglify', 'our', 'code']
print(replacer.replace_negations(sent))

class AntonymWordReplacer(WordReplacer, AntonymReplacer):
    pass

# Example using a dictionary and replacing negations
replacer = AntonymWordReplacer({'evil': 'good'})
result = replacer.replace_negations(['good', 'is', 'not', 'evil'])
print(result)
