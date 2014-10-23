#dependency: tweet_processingV3.py
################################################################################################
################################################################################################
#process text for sentiments scores
################################################################################################
################################################################################################
#import sentiwordnet
try:
    from nltk.corpus import sentiwordnet as swn
except:
    import sys
    #work computer directory
#    dir1='/home/sgolbeck/nltk_data/corpora/sentiwordnet'
    #home computer directory
    dir1='/home/golbeck/nltk_data/corpora/sentiwordnet'
    sys.path.append(dir1)
    print sys.path
    from sentiwordnet import SentiWordNetCorpusReader, SentiSynset
    #file that contains the polarity and objectivity scores
    swn_filename = dir1+"/SentiWordNet_3.0.0.txt"
    #swn_filename = "SentiWordNet_3.0.0_20100705.txt"
    swn = SentiWordNetCorpusReader(swn_filename)
    
#test if sentiwordnet has been imported
swn_bad=swn.senti_synsets('bad')

#######################################################
#to use the below, copy the SentiWordNet*.txt file into pattern/en/wordnet
from pattern.en import wordnet
#EXAMPLE: 
print wordnet.synsets("kill",pos="VB")[0].weight
from pattern.en import ADJECTIVE
#EXAMPLE: 
pattern_bad=wordnet.synsets('bad', ADJECTIVE)[0]
#######################################################
from pattern.en import parse
#EXAMPLE: 
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
pattern_bad_parse=pattern_bad_parse.split()
print pattern_bad_parse
pattern_bad_parse_word=pattern_bad_parse[0][3]

        
#######################################################
import nltk
#EXAMPLE: 
text=nltk.word_tokenize("And now for something completely different")
#requires that 'maxent_treebank_pos_tagger' and 'punkt' has been downloaded
text_tagged=nltk.pos_tag(text)
##############################################################################################################
##############################################################################################################
##############################################################################################################
#FUNCTIONS FOR SENTIMENT 
##############################################################################################################
##############################################################################################################
##############################################################################################################
from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
#    {'NN':'n', 'VB':'v', 'JJ':'a', 'RB':'r'}
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
#######################################################

def get_sentiment(tweet_text,i):
    #input: list of tokens and an integer indicating an element of the list
    #output: tuple of (pos sentiment,neg sentiment)
    try:
        sent_out=swn.senti_synsets(tweet_text[0][i][0],get_wordnet_pos(tweet_text[0][i][1]))
        if len(sent_out)>0:
            sent_out=sent_out[0]
            #change to pos_score() and neg_score() if on home computer
            #else, pos_score and neg_score if on work computer
            swn_score=(sent_out.pos_score(),sent_out.neg_score())
        else:
            swn_score=(0.0,0.0)
    except:
        swn_score=(0.0,0.0)
        
    return swn_score
#######################################################

def swn_sentiment(token_list):
    #input: a list of tokens
    #output: list of [pos sentiment,neg sentiment]
    #uses sentiwordnet, requires parse from pattern
    #convert tokens to string
    tweet_text = " ".join(token_list)
    #remove ticker symbols so as they are assigned 0 sentiment score
    tweet_text = tweet_text.replace('$','TICKSYMBOL_')
    #remove hashtags so that they receive sentiment scores
    tweet_text = tweet_text.replace('#','')
    #tag each element in the string with a POS, then split into a list
    tweet_text=parse(tweet_text).split()
    N=len(tweet_text[0])
    #assign sentiment to each element of the parsed string
    sentiment_list=[get_sentiment(tweet_text,i) for i in range(N)]
    #sum up the sentiment scores (pos and neg) for each tweet
    senti_out=[sum([sentiment_list[x][i] for x in range(len(sentiment_list))]) for i in range(2)]
    return senti_out
##############################################################################################################
##############################################################################################################
##############################################################################################################
#EXAMPLE: 
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
temp=pattern_bad_parse.split()[0]
for i in range(len(temp)):
    try:
        print swn.senti_synsets(temp[i][0],get_wordnet_pos(temp[i][1]))[0]
    except:
        print "no entry in sentiwordnet"

#######################################################
#EXAMPLE: test on an actual tweet in the DataFrame
#list of tokens
tweet_text_list=DF.ix[0,7]
#convert tokens to string
tweet_test_str = " ".join(tweet_text_list)
#remove ticker symbols so as they are assigned 0 sentiment score
tweet_test_str = tweet_test_str.replace('$','TICKSYMBOL_')
#remove hashtags so that they receive sentiment scores
tweet_test_str = tweet_test_str.replace('#','')
#tag each element in the string with a POS, then split into a list
tweet_test_str=parse(tweet_test_str).split()

for i in range(len(tweet_test_str[0])):
    try:
        print swn.senti_synsets(tweet_test_str[0][i][0],get_wordnet_pos(tweet_test_str[0][i][1]))[0]
    except:
        print "no entry in sentiwordnet"


#######################################################
#score the sentiment of each tweet that references a ticker
senti_list=[]
senti_list=[swn_sentiment(DF_tick.ix[i,7]) for i in DF_tick.index]
#save the sentiment scores in the dataframe
DF_tick[12]=senti_list

