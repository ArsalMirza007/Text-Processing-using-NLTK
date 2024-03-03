import re

replacement_patterns = [
    (r"won't", 'will not'),
    (r"can't", 'cannot'),
    (r"i'm", 'i am'),
    (r"ain't", 'is not'),
    (r"(\w+)'ll", r'\1 will'),
    (r"(\w+)n't", r'\1 not'),
    (r"(\w+)'ve", r'\1 have'),
    (r"(\w+)'s", r'\1 is'),
    (r"(\w+)'re", r'\1 are'),
    (r"(\w+)'d", r'\1 would')
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

replacer = RegexpReplacer()

text = "can't is a contraction"
replaced_text = replacer.replace(text)
print(replaced_text)

from nltk.tokenize import word_tokenize
replaced_tokens = word_tokenize(replaced_text)
print(replaced_tokens)
