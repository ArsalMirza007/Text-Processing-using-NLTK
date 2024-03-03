from nltk.chunk import RegexpParser
from nltk.chunk.regexp import ChunkString, ChunkRule, MergeRule, SplitRule, RegexpChunkRule
from nltk.tree import Tree

# Define a chunker with regular expressions
chunker = RegexpParser(r'''
    NP:
        {<DT><.*>*<NN.*>}
        <NN.*>}{<.*>
        <.*>}{<DT>
        <NN.*>{}<NN.*>
''')

# Define a sentence
sent = [('the', 'DT'), ('sushi', 'NN'), ('roll', 'NN'), ('was', 'VBD'), ('filled', 'VBN'), ('with', 'IN'), ('the', 'DT'), ('fish', 'NN')]

# Parse the sentence using the chunker
result = chunker.parse(sent)
print(result)

# Create a ChunkString
cs = ChunkString(Tree('S', sent))
print(cs)

# Apply ChunkRule to chunk determiner to noun
ur = ChunkRule('<DT><.*>*<NN.*>', 'chunk determiner to noun')
ur.apply(cs)
print(cs)

# Apply SplitRule to split after noun
sr1 = SplitRule('<NN.*>', '<.*>', 'split after noun')
sr1.apply(cs)
print(cs)

# Apply SplitRule to split before determiner
sr2 = SplitRule('<.*>', '<DT>', 'split before determiner')
sr2.apply(cs)
print(cs)

# Apply MergeRule to merge nouns
mr = MergeRule('<NN.*>', '<NN.*>', 'merge nouns')
mr.apply(cs)
print(cs)

# Convert ChunkString to ChunkStruct
chunkstruct = cs.to_chunkstruct()
print(chunkstruct)

# Create ChunkRules from strings
rule1 = RegexpChunkRule.fromstring('{<DT><.*>*<NN.*>}')
print(rule1)

rule2 = RegexpChunkRule.fromstring('<.*>}{<DT>')
print(rule2)

rule3 = RegexpChunkRule.fromstring('<NN.*>{}<NN.*>')
print(rule3)

# Get the description of a rule
print(rule1.descr)
print(rule2.descr)

