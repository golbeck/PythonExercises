#if in the twitter directory, change to Tweets directory
#################################################

import nltk
from pandas import DataFrame, Series
import pandas as pd
from nltk import word_tokenize
pattern = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \$\w+  # currency and percentages, e.g. $12.40, 82%
    | @\S+             # tokenize twitter mentions
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
    '''

#################################################
#import all of the tweets from the .csv files
import os
pwd_temp=%pwd
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Tweets'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Tweets'
if pwd_temp!=dir1:
    os.chdir(dir1)

import glob
tweet_list=[]
tab_errors=[]
path = "*.csv"
DF=DataFrame()
for fname in glob.glob(path):
    temp_text=nltk.regexp_tokenize(fname, pattern)
    name_temp=temp_text[0]
    tweet_list.append(name_temp.lower())
    try:
        DF1=pd.io.parsers.read_table(fname,sep=',',index_col=0,header=None)
        index_max=len(DF)
        DF1.index=range(index_max,index_max+len(DF1))
        DF=DF.append(DF1)
        print 'successfully appended ' +fname
    except:
        print 'there was an error: '+fname
        tab_errors.append(fname)
        continue

#delete the rows associated with tick_indices.csv and any NaN entries (>1)
DF=DF.dropna(thresh=2)
DF.index=range(len(DF))

#sort tweet_list
tweet_list=list(set(tweet_list))
tweet_list.sort()
################################################################################################
################################################################################################
#columns of DF
##  1: date
##  2: day of week
##  3: followers
##  4: tweet identification number
##  5: retweets
##  6: user id
##  7: list of tokens
##  8: list of stock tickers
##  9: time of day hour:min:sec
## 10: user id number
## 11: datetime object of format (year,month,day,hour,minute,second) created below
################################################################################################
################################################################################################
###convert string in DataFrame 'DF' to a list
#################################################
#do this for the tokenized tweet
temp_list=[]
for i in range(len(DF)):
    temp=DF.ix[i,7].split('\'')[1:-1]
    temp1=[]
    [temp1.append(x) for x in temp if x not in [', ']]
    temp_list.append(temp1)
#replace strings in column with a list of lists
DF[7]=temp_list
#################################################
#do this for the list of tickers
temp_list=[]
#array of indices for rows in DF that contain a ticker symbol
tick_index=[]
#generate a list of actual ticker symbols
tick_list=[]
for i in range(len(DF)):
    temp=DF.ix[i,8].split('\'')[1:-1]
    temp1=[]
    [temp1.append(x) for x in temp if x not in [', ']]
    temp_list.append(temp1)
    if len(temp1)>0:
        tick_index.append(i)
        [tick_list.append(x) for x in set(temp1)]
#replace strings in column with a list of lists
DF[8]=temp_list
#################################################
#select tweets from a certain period
#see https://docs.python.org/2/library/datetime.html#datetime-objects
#and https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
from datetime import datetime
DF_dates=[]
for i in range(len(DF)):
    date_object = datetime.strptime(DF.ix[i,1], '%Y-%m-%d')
    date_object1 = datetime.strptime(DF.ix[i,9], '%H:%M:%S')
    date_object = date_object.replace(hour=date_object1.hour)
    date_object = date_object.replace(minute=date_object1.minute)
    date_object = date_object.replace(second=date_object1.second)
    DF_dates.append(date_object)
#create a new column in 'DF' of datetime objects
DF[11]=DF_dates
################################################################################################
################################################################################################
#select only rows that reference a particular ticker symbol
DF_tick=DF.ix[tick_index,:]
#reindex the dataframe (uncomment line below)
#DF_tick.index=range(len(DF_tick))
#unique list of ticker symbols in tweets
tick_list=list(set(tick_list))
#################################################
#count how many occurences of each ticker symbol
tick_count={}
for x in tick_list:
    counter=[1 for j in DF_tick[8] if x in j]
    tick_count[x]=sum(counter)
#################################################
#create a dataframe of ticker counts
DF_tick_count=DataFrame.from_dict(tick_count,orient='index')
#sort ticker counts
DF_tick_count=DF_tick_count.sort(columns=0,ascending=False)
#import all of the tweet indices (for each ticker) from the .csv file

#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Tweets'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Tweets'
if pwd_temp!=dir1:
    os.chdir(dir1)
DF_tick_indices=pd.io.parsers.read_table('tick_indices.csv',sep=',',header=None)

#generate the dictionary of tweet indices for each ticker
tick_indices={}
for i in range(len(DF_tick_indices)):
    temp=DF_tick_indices.ix[i,1][1:-1].split(', ')
    temp1=[]
    [temp1.append(int(x)) for x in temp]
    tick_indices[DF_tick_indices.ix[i,0]]=temp1
################################################################################################
################################################################################################
#process text for sentiments scores
################################################################################################
################################################################################################
try:
    from nltk.corpus import sentiwordnet as swn
except:
    import sys
    #work computer directory
    dir1='/home/sgolbeck/nltk_data/corpora/sentiwordnet'
    #home computer directory
#    dir1='/home/golbeck/nltk_data/corpora/sentiwordnet'
    sys.path.append(dir1)
    print sys.path
    from sentiwordnet import SentiWordNetCorpusReader, SentiSynset
    swn_filename = dir1+"/SentiWordNet_3.0.0.txt"
    #swn_filename = "SentiWordNet_3.0.0_20100705.txt"
    swn = SentiWordNetCorpusReader(swn_filename)

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


#######################################################
#import pandas as pd
#DF=pd.io.parsers.read_table('SentiWordNet_3.0.1.txt',sep='\t',header=26)
#DF.ix[[x for x in range(len(DF)) if DF['SynsetTerms'][x].startswith('bad')==True and DF['POS'][x]=='a'],:]

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
            swn_score=(sent_out.pos_score,sent_out.neg_score)
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


##################################################################################################
##################################################################################################
##################################################################################################
#create a time window for August
left_window=datetime(2014,8,1,0,0,0)
right_window=datetime(2014,9,1,0,0,0)
#test if date_object is in the window
cond_l=(DF_tick[11]>=left_window)
cond_r=(DF_tick[11]<right_window)
cond=[cond_l[i] and cond_r[i] for i in DF_tick.index]
#select only those rows within the window
DF_aug=DF_tick[cond]
#################################################
#count how many occurences of each ticker symbol
tick_count_aug={}
for x in tick_list:
    counter=[1 for j in DF_aug[8] if x in j]
    tick_count_aug[x]=sum(counter)
#################################################
#create a dataframe of ticker counts
DF_tick_count_aug=DataFrame.from_dict(tick_count_aug,orient='index')
#sort ticker counts
DF_tick_count_aug=DF_tick_count_aug.sort(columns=0,ascending=False)
#################################################
#EXAMPLE: process tweet sent
x='AAPL'
tweet_indices=[i for i in DF_aug.index if i in tick_indices[x]]
temp=DF_aug.ix[tweet_indices]

#only examine tweets that reference a single stock ticker
tick_indices_unique=[]
for i in DF_tick.index:
    if len(DF_tick.ix[i,8])==1:
        tick_indices_unique.append(i)

#intersection of august tweets with tickers & tweets that reference a single ticker
tick_indices_unique_aug=[i for i in DF_aug.index if i in tick_indices_unique]

#determine average pos and neg sentiment for each ticker
#restrict dataframe to only tickers with at least m tweets
m=20
DF_tick_count_aug_m=DF_tick_count_aug[DF_tick_count_aug[0]>m]

sent_pos_aug={}
sent_neg_aug={}
for x in DF_tick_count_aug_m:
    sent_pos=0.0
    sent_neg=0.0
    for i in [j for j in tick_indices_unique if j in tick_indices[x]]:
        sent_pos=sent_pos+DF_aug.ix[i,12][0]
        sent_neg=sent_neg+DF_aug.ix[i,12][1]
    sent_pos_aug[x]=sent_pos/float(DF_tick_count_aug_m.ix[x,0])
    sent_neg_aug[x]=sent_neg/float(DF_tick_count_aug_m.ix[x,0])




