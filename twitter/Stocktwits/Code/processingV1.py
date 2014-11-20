#if in the twitter directory, change to Tweets directory
#################################################
import re
import nltk
import os
import csv
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


##################################################################################################
##################################################################################################
##################################################################################################
def contraction(text):
    n=len(text)
    if n>1:
        for i in range(1,n):
            if len(text[i])==3:
                #not
                if text[i][0:3]=='39t':
                    if text[i-1]=='aren':
                        text[i-1]='are'
                        text[i]='not'
                    if text[i-1]=='can':
                        text[i-1]='can'
                        text[i]='not'
                    if text[i-1]=='couldn':
                        text[i-1]='could'
                        text[i]='not'
                    if text[i-1]=='didn':
                        text[i-1]='did'
                        text[i]='not'
                    if text[i-1]=='doesn':
                        text[i-1]='does'
                        text[i]='not'
                    if text[i-1]=='don':
                        text[i-1]='do'
                        text[i]='not'
                    if text[i-1]=='hadn':
                        text[i-1]='had'
                        text[i]='not'
                    if text[i-1]=='hasn':
                        text[i-1]='has'
                        text[i]='not'
                    if text[i-1]=='haven':
                        text[i-1]='have'
                        text[i]='not'
                    if text[i-1]=='isn':
                        text[i-1]='is'
                        text[i]='not'
                    if text[i-1]=='mightn':
                        text[i-1]='might'
                        text[i]='not'
                    if text[i-1]=='mustn':
                        text[i-1]='must'
                        text[i]='not'
                    if text[i-1]=='shan':
                        text[i-1]='shall'
                        text[i]='not'
                    if text[i-1]=='shouldn':
                        text[i-1]='should'
                        text[i]='not'
                    if text[i-1]=='weren':
                        text[i-1]='were'
                        text[i]='not'
                    if text[i-1]=='won':
                        text[i-1]='will'
                        text[i]='not'
                    if text[i-1]=='wouldn':
                        text[i-1]='would'
                        text[i]='not'
                #is
                if text[i][0:3]=='39s':
                    if text[i-1]=='it':
                        text[i-1]='it'
                        text[i]='is'
                    if text[i-1]=='he':
                        text[i-1]='he'
                        text[i]='is'
                    if text[i-1]=='she':
                        text[i-1]='she'
                        text[i]='is'
                    if text[i-1]=='that':
                        text[i-1]='that'
                        text[i]='is'
                    if text[i-1]=='there':
                        text[i-1]='there'
                        text[i]='is'
                    if text[i-1]=='what':
                        text[i-1]='what'
                        text[i]='is'
                    if text[i-1]=='where':
                        text[i-1]='where'
                        text[i]='is'
                    if text[i-1]=='who':
                        text[i-1]='who'
                        text[i]='is'

                #am
                if text[i][0:3]=='39m':
                    if text[i-1]=='I':
                        text[i-1]='I'
                        text[i]='am'
                #would
                if text[i][0:3]=='39d':
                    if text[i-1]=='he':
                        text[i-1]='he'
                        text[i]='would'
                    if text[i-1]=='I':
                        text[i-1]='I'
                        text[i]='would'
                    if text[i-1]=='she':
                        text[i-1]='she'
                        text[i]='would'
                    if text[i-1]=='they':
                        text[i-1]='they'
                        text[i]='would'
                    if text[i-1]=='we':
                        text[i-1]='we'
                        text[i]='would'
                    if text[i-1]=='who':
                        text[i-1]='who'
                        text[i]='would'
                    if text[i-1]=='you':
                        text[i-1]='you'
                        text[i]='would'
                #am
                if text[i][0:3]=='39m':
                    if text[i-1]=='I':
                        text[i-1]='I'
                        text[i]='am'
            if len(text[i])==4:
                #will/shall
                if text[i][0:4]=='39ll':
                    if text[i-1]=='she':
                        text[i-1]='she'
                        text[i]='will'
                    if text[i-1]=='he':
                        text[i-1]='he'
                        text[i]='will'
                    if text[i-1]=='I':
                        text[i-1]='I'
                        text[i]='will'
                    if text[i-1]=='they':
                        text[i-1]='they'
                        text[i]='will'
                    if text[i-1]=='what':
                        text[i-1]='what'
                        text[i]='will'
                    if text[i-1]=='who':
                        text[i-1]='who'
                        text[i]='will'
                    if text[i-1]=='you':
                        text[i-1]='you'
                        text[i]='will'
                #have
                if text[i][0:4]=='39ve':
                    if text[i-1]=='I':
                        text[i-1]='I'
                        text[i]='have'
                    if text[i-1]=='they':
                        text[i-1]='they'
                        text[i]='have'
                    if text[i-1]=='we':
                        text[i-1]='we'
                        text[i]='have'
                    if text[i-1]=='what':
                        text[i-1]='what'
                        text[i]='have'
                    if text[i-1]=='who':
                        text[i-1]='who'
                        text[i]='have'
                    if text[i-1]=='you':
                        text[i-1]='you'
                        text[i]='have'
                #are
                if text[i][0:4]=='39re':
                    if text[i-1]=='they':
                        text[i-1]='they'
                        text[i]='are'
                    if text[i-1]=='we':
                        text[i-1]='we'
                        text[i]='are'
                    if text[i-1]=='what':
                        text[i-1]='what'
                        text[i]='are'
                    if text[i-1]=='who':
                        text[i-1]='who'
                        text[i]='are'
                    if text[i-1]=='you':
                        text[i-1]='you'
                        text[i]='are'
    return text

#generate list of tickers for stocks
pwd_temp=%pwd
#work computer directory
dir1_='/home/sgolbeck/workspace/'
#home computer directory
#dir1_='/home/sgolbeck/Workspace/
dir1=dir1_+'PythonExercises/twitter'

if pwd_temp!=dir1:
    os.chdir(dir1)
ticklist1=pd.io.parsers.read_table('Ticker list/AMEX.csv',sep=',')
ticklist2=pd.io.parsers.read_table('Ticker list/NYSE.csv',sep=',')
ticklist3=pd.io.parsers.read_table('Ticker list/NASDAQ.csv',sep=',')
ticklist=ticklist1.merge(ticklist2,how='outer')
ticklist=ticklist.merge(ticklist3,how='outer')
tickers1=ticklist[ticklist['MarketCap']>0.0]
tickers2=list(tickers1['Symbol'])
#tickers3=[w.lower() for w in tickers2]
tickers3=[re.sub(r'\s','',w) for w in tickers2]
tickers=set(tickers3)

#################################################
#import all of the tweets from the .csv files
pwd_temp=%pwd
dir1=dir1_+'PythonExercises/twitter/Stocktwits'
if pwd_temp!=dir1:
    os.chdir(dir1)

import glob
tab_errors=[]
path = "*.csv"
DF=DataFrame()
for fname in sorted(glob.glob(path)):
    try:
        DF1=pd.io.parsers.read_table(fname,sep=',',index_col=0,header=0)
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
    temp=DF.ix[i,'text'].split('\'')[1:-1]
    temp1=[]
    [temp1.append(x) for x in temp if x not in [', ']]
    temp_list.append(temp1)
#replace strings in column with a list of lists
DF['text']=temp_list
#################################################
ticks_list=[]
for i in range(len(DF)):
    ticks=[w.upper() for w in DF.ix[i,'text'] if w.startswith('$')]
    if len(ticks)>0.0:
        ticks=[w[1:] for w in ticks if w[1].isalpha()]
    ticks_list.append(ticks)
DF['symbol']=ticks_list
#################################################
#drop rows that are duplicated
DF=DF.drop_duplicates(['message_id'])

DF=DF.sort(columns=['message_id'],ascending=True)
DF.index=range(len(DF))

#change '39_' tokens to contractions
t1=[]
for i in range(len(DF)):
    t1.append(contraction(DF.ix[i,'text']))
DF['text']=t1

#save the dataframe to a CSV file
dir1=dir1_+'PythonExercises/twitter/Stocktwits/Processed'
if pwd_temp!=dir1:
    os.chdir(dir1)
writer = csv.writer(open('Stocktwits_database.csv', 'wb'))
for i in DF.index:
    writer.writerow([DF.ix[i,j] for j in DF.columns])

##################################################################################################
##################################################################################################
##################################################################################################
#dataframe with only tweets with one ticker
tick_ind=[]
[tick_ind.append(j) for j in range(len(DF)) if len(DF.ix[j,'symbol'])==1]
DF_single=DF.ix[tick_ind]
DF_single.index=range(len(DF_single))


#save the dataframe to a CSV file
dir1=dir1_+'PythonExercises/twitter/Stocktwits/Processed'
if pwd_temp!=dir1:
    os.chdir(dir1)
writer = csv.writer(open('Stocktwits_database_single.csv', 'wb'))
for i in DF_single.index:
    writer.writerow([DF_single.ix[i,j] for j in DF_single.columns])

#generate a random sample for training the data
import random
ind_train=random.sample(DF_single.index,1000)
DF_train=DF_single.ix[ind_train]
#save the training set to CSV
dir1=dir1_+'PythonExercises/twitter/Stocktwits/Processed'
if pwd_temp!=dir1:
    os.chdir(dir1)
writer = csv.writer(open('Stocktwits_database_train.csv', 'wb'))
for i in DF_train.index:
    writer.writerow([DF_train.ix[i,j] for j in DF_train.columns])
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
def ticker_increment(tick_indices,key,ind):
    #builds dictionary of indices of the dataframe associated with each ticker symbol
    if key in tick_indices.keys():
        tick_indices[key].append(ind)
    else:
        tick_indices[key]=[ind]
    return 0.0
#################################################
#list of indices corresponding to each ticker symbol
#(try to redo this with map-reduce)
tick_indices={}
for j in range(len(DF)):
    print j
    print ' out of '
    print len(DF)
    [ticker_increment(tick_indices,x,j) for x in DF.ix[j,'symbol']]
#################################################
##save the ticker indices to a csv file in the Processed directory
##work computer directory
#dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Stocktwits/Processed'
##home computer directory
##dir1='/home/sgolbeck/Workspace/PythonExercises/twitter/Stocktwits/Processed'
#if pwd_temp!=dir1:
#    os.chdir(dir1)
#writer = csv.writer(open('tick_indices.csv', 'wb'))
#for key in tick_indices.keys():
#    writer.writerow([key, tick_indices[key]])
#################################################
#count how many occurences of each ticker symbol
tick_count={}
for x in tick_indices.keys():
    tick_count[x]=len(tick_indices[x])
#################################################
#create a dataframe of ticker counts
DF_tick_count=DataFrame.from_dict(tick_count,orient='index')
#sort ticker counts
DF_tick_count=DF_tick_count.sort(columns=0,ascending=False)
DF_tick_count.columns=['count']


##################################################################################################
##################################################################################################
##################################################################################################
#only tweets that reference a ticker listed on a major exchange
#dataframe with only tweets with one ticker
tick_ind=[]
[tick_ind.append(j) for j in range(len(DF)) if len(DF.ix[j,'symbol'])==1]
DF_single=DF.ix[tick_ind]
DF_single.index=range(len(DF_single))
tick_ind=[]
[tick_ind.append(j) for j in range(len(DF_single)) if DF_single.ix[j,'symbol'][0] in tickers]
DF_single_public=DF_single.ix[tick_ind]
DF_single_public.index=range(len(DF_single_public))

#save the dataframe to a CSV file
dir1=dir1_+'PythonExercises/twitter/Stocktwits/Processed'
if pwd_temp!=dir1:
    os.chdir(dir1)
writer = csv.writer(open('Stocktwits_database_single_public.csv', 'wb'))
for i in DF_single_public.index:
    writer.writerow([DF_single_public.ix[i,j] for j in DF_single_public.columns])

#remove Swing Traders from dataframe
DF_single_swing=DF_single[DF_single['holding_per']!='Swing Trader']
DF_single_position=DF_single[DF_single['holding_per']!='Position Trader']
DF_single_position_swing=DF_single_position[DF_single_position['holding_per']!='Swing Trader']



##################################################################################################
##################################################################################################
##################################################################################################
#find bigrams and sort according to frequency
#generate n-gram frequencies from DF_single_public
#first, combine all tweets into a single corpus, separating each with a period
token_list=[]
for i in range(len(DF_single_public)):
    [token_list.append(x.lower()) for x in DF_single_public.ix[i,'text']]
    #separate tweets by periods
    token_list.append('.')
print len(token_list)

#import collocation packages
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

#The collocations package provides collocation finders which by default consider all ngrams in a text as candidate collocations:
finder = BigramCollocationFinder.from_words(token_list)
#raw frequency scoring, can choose other methods
scored = finder.score_ngrams(bigram_measures.raw_freq)
bigram_scores=[x for x in scored]
#sorted(bigram for bigram, score in scored)

##manual frequency scoring
#word_fd = nltk.FreqDist(token_list)
#bigram_fd = nltk.FreqDist(nltk.bigrams(token_list))
#finder = BigramCollocationFinder(word_fd, bigram_fd)
#scored == finder.score_ngrams(bigram_measures.raw_freq)
#sorted(bigram for bigram, score in scored)



##################################################################################################
##################################################################################################
##################################################################################################
#requires that 'maxent_treebank_pos_tagger' and 'punkt' has been downloaded
#for tags see:
#   http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
#   nltk.help.upenn_tagset()
text=DF_single_public.ix[0,'text']
text_tagged=nltk.pos_tag(text)
#an NP chunk should be formed whenever the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) and then a noun (NN). 
#see http://www.nltk.org/book/ch07.html
#using regular expression:
#    https://docs.python.org/2/library/re.html
grammar = "NP: {<DT>?<JJ>*<NN>}"
grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
#create a chunk parser
cp = nltk.RegexpParser(grammar)
#test it on our example sentence
result = cp.parse(text_tagged)
print(result)
