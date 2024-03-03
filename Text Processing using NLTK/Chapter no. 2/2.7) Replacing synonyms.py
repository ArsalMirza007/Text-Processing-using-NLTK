import csv
import yaml

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)

# Example using a dictionary
replacer = WordReplacer({'bday': 'birthday'})
print(replacer.replace('bday'))  # Output: 'birthday'
print(replacer.replace('happy'))  # Output: 'happy'

class CsvWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = {}
        with open(fname, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                word, syn = line
                word_map[word] = syn
        super(CsvWordReplacer, self).__init__(word_map)

# Example using a CSV file
csv_replacer = CsvWordReplacer('synonyms.csv')
print(csv_replacer.replace('bday'))  # Output: 'birthday'
print(csv_replacer.replace('happy'))  # Output: 'happy'

class YamlWordReplacer(WordReplacer):
    def __init__(self, fname):
        word_map = yaml.load(open(fname, 'r'))
        super(YamlWordReplacer, self).__init__(word_map)

# Example using a YAML file
yaml_replacer = YamlWordReplacer('synonyms.yaml')
print(yaml_replacer.replace('bday'))  # Output: 'birthday'
print(yaml_replacer.replace('happy'))  # Output: 'happy'
