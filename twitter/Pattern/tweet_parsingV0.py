#######################################################
cd /home/sgolbeck/nltk_data/corpora/sentiwordnet
from sentiwordnet import SentiWordNetCorpusReader, SentiSynset
swn_filename = "SentiWordNet_3.0.0.txt"
#swn_filename = "SentiWordNet_3.0.0_20100705.txt"
swn = SentiWordNetCorpusReader(swn_filename)
swn_bad=swn.senti_synsets('bad')

#######################################################
from pattern.en import wordnet
print wordnet.synsets("kill",pos="VB")[0].weight
from pattern.en import ADJECTIVE
pattern_bad=wordnet.synsets('bad', ADJECTIVE)[0]
#######################################################
from pattern.en import parse
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
pattern_bad_parse=pattern_bad_parse.split()
print pattern_bad_parse
pattern_bad_parse_word=patter_bad_parse[3]

#post_dict={'NN':'n', 'VB':'v', 'JJ':'a', 'RB':'r'}

#######################################################
import nltk
text=nltk.word_tokenize("And now for something completely different")
#requires that 'maxent_treebank_pos_tagger' has been downloaded
text_tagged=nltk.pos_tag(text)


#######################################################
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
        
        
DF=pandas.io.parsers.read_table('SentiWordNet_3.0.1.txt',sep='\t',header=26)
DF.ix[[x for x in range(len(DF)) if DF['SynsetTerms'][x].startswith('bad')==True and DF['POS'][x]=='a'],:]

from nltk.corpus import wordnet
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
temp=pattern_bad_parse.split()[0]
for i in range(len(temp)):
    try:
        print swn.senti_synsets(temp[i][0],get_wordnet_pos(temp[i][1]))[0]
    except:
        print "no entry in sentiwordnet"
