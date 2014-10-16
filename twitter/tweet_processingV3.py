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
dir1='/home/golbeck/Workspace/PythonExercises/twitter/Tweets'
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
dir1='/home/golbeck/Workspace/PythonExercises/twitter/Tweets'
if pwd_temp!=dir1:
    os.chdir(dir1)
DF_tick_indices=pd.io.parsers.read_table('tick_indices.csv',sep=',',header=None)
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
    dir1='/home/golbeck/nltk_data/corpora/sentiwordnet'
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
print wordnet.synsets("kill",pos="VB")[0].weight
from pattern.en import ADJECTIVE
pattern_bad=wordnet.synsets('bad', ADJECTIVE)[0]
#######################################################
from pattern.en import parse
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
pattern_bad_parse=pattern_bad_parse.split()
print pattern_bad_parse
pattern_bad_parse_word=pattern_bad_parse[0][3]

        
#######################################################
import nltk
text=nltk.word_tokenize("And now for something completely different")
#requires that 'maxent_treebank_pos_tagger' and 'punkt' has been downloaded
text_tagged=nltk.pos_tag(text)


#######################################################
#import pandas as pd
#DF=pd.io.parsers.read_table('SentiWordNet_3.0.1.txt',sep='\t',header=26)
#DF.ix[[x for x in range(len(DF)) if DF['SynsetTerms'][x].startswith('bad')==True and DF['POS'][x]=='a'],:]

#######################################################
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
pattern_bad_parse=parse('he is a bad man of crime that dances violently')
temp=pattern_bad_parse.split()[0]
for i in range(len(temp)):
    try:
        print swn.senti_synsets(temp[i][0],get_wordnet_pos(temp[i][1]))[0]
    except:
        print "no entry in sentiwordnet"

#test on an actual tweet in the DataFrame
tweet_text_list=DF.ix[0,7]
tweet_test_str = " ".join(tweet_text_list)
tweet_test_str=parse('he is a bad man of crime that dances violently').split()
for i in range(len(tweet_test_str)):
    try:
        print swn.senti_synsets(tweet_test_str[i][0],get_wordnet_pos(tweet_test_str[i][1]))[0]
    except:
        print "no entry in sentiwordnet"
