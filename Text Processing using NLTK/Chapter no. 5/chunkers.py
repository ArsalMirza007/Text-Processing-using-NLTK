import nltk
nltk.download('names')
nltk.download('ieer')
nltk.download('gazetteers')
import nltk.tag
from nltk.chunk import ChunkParserI
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import UnigramTagger, BigramTagger, ClassifierBasedTagger
from nltk.corpus import names, ieer, gazetteers

# Stub for the missing 'tag_util' module
def backoff_tagger(train_sents, tagger_classes):
    default_tagger = nltk.DefaultTagger('NN')
    return nltk.UnigramTagger(train_sents, backoff=default_tagger)

def conll_tag_chunks(chunk_sents):
    tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

class TagChunker(ChunkParserI):
    def __init__(self, train_chunks, tagger_classes=[UnigramTagger, BigramTagger]):
        train_sents = conll_tag_chunks(train_chunks)
        self.tagger = backoff_tagger(train_sents, tagger_classes)
    
    def parse(self, tagged_sent):
        if not tagged_sent:
            return None
        (words, tags) = zip(*tagged_sent)
        chunks = self.tagger.tag(tags)
        wtc = zip(words, chunks)
        return conlltags2tree([(w, t, c) for (w, (t, c)) in wtc])

def chunk_trees2train_chunks(chunk_sents):
    tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
    return [[((w, t), c) for (w, t, c) in sent] for sent in tag_sents]

def prev_next_pos_iob(tokens, index, history):
    word, pos = tokens[index]
    
    if index == 0:
        prevword, prevpos, previob = ('<START>',) * 3
    else:
        prevword, prevpos = tokens[index - 1]
        previob = history[index - 1]
    
    if index == len(tokens) - 1:
        nextword, nextpos = ('<END>',) * 2
    else:
        nextword, nextpos = tokens[index + 1]
    
    feats = {
        'word': word,
        'pos': pos,
        'nextword': nextword,
        'nextpos': nextpos,
        'prevword': prevword,
        'prevpos': prevpos,
        'previob': previob
    }
    
    return feats

class ClassifierChunker(ChunkParserI):
    def __init__(self, train_sents, feature_detector=prev_next_pos_iob, **kwargs):
        if not feature_detector:
            feature_detector = self.feature_detector
        
        train_chunks = chunk_trees2train_chunks(train_sents)
        self.tagger = ClassifierBasedTagger(train=train_chunks, feature_detector=feature_detector, **kwargs)
    
    def parse(self, tagged_sent):
        if not tagged_sent:
            return None
        chunks = self.tagger.tag(tagged_sent)
        return conlltags2tree([(w, t, c) for ((w, t), c) in chunks])

def sub_leaves(tree, label):
    return [t.leaves() for t in tree.subtrees(lambda s: s.label() == label)]

class PersonChunker(ChunkParserI):
    def __init__(self):
        self.name_set = set(names.words())
    
    def parse(self, tagged_sent):
        iobs = []
        in_person = False
        
        for word, tag in tagged_sent:
            if word in self.name_set and in_person:
                iobs.append((word, tag, 'I-PERSON'))
            elif word in self.name_set:
                iobs.append((word, tag, 'B-PERSON'))
                in_person = True
            else:
                iobs.append((word, tag, 'O'))
                in_person = False
        
        return conlltags2tree(iobs)

class LocationChunker(ChunkParserI):
    def __init__(self):
        self.locations = set(gazetteers.words())
        self.lookahead = 0
        
        for loc in self.locations:
            nwords = loc.count(' ')
            
            if nwords > self.lookahead:
                self.lookahead = nwords
    
    def iob_locations(self, tagged_sent):
        i = 0
        l = len(tagged_sent)
        inside = False
        
        while i < l:
            word, tag = tagged_sent[i]
            j = i + 1
            k = j + self.lookahead
            nextwords, nexttags = [], []
            loc = False
            
            while j < k:
                if ' '.join([word] + nextwords) in self.locations:
                    if inside:
                        yield word, tag, 'I-LOCATION'
                    else:
                        yield word, tag, 'B-LOCATION'
                    
                    for nword, ntag in zip(nextwords, nexttags):
                        yield nword, ntag, 'I-LOCATION'
                    
                    loc, inside = True, True
                    i = j
                    break
                
                if j < l:
                    nextword, nexttag = tagged_sent[j]
                    nextwords.append(nextword)
                    nexttags.append(nexttag)
                    j += 1
                else:
                    break
            
            if not loc:
                inside = False
                i += 1
                yield word, tag, 'O'
    
    def parse(self, tagged_sent):
        iobs = self.iob_locations(tagged_sent)
        return conlltags2tree(iobs)

def ieertree2conlltags(tree, tag=nltk.tag.pos_tag):
    words, ents = zip(*tree.pos())
    iobs = []
    prev = None
    
    for ent in ents:
        if ent == tree.label():
            iobs.append('O')
            prev = None
        elif prev == ent:
            iobs.append('I-%s' % ent)
        else:
            iobs.append('B-%s' % ent)
            prev = ent
    
    words, tags = zip(*tag(words))
    return zip(words, tags, iobs)

def ieer_chunked_sents(tag=nltk.tag.pos_tag):
    for doc in ieer.parsed_docs():
        tagged = ieertree2conlltags(doc.text, tag)
        yield conlltags2tree(tagged)
        
if __name__ == '__main__':
    import doctest
    result = doctest.testmod()
    print("Doctest Results:")
    print(result)
