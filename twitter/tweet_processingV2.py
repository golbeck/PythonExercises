#if in the twitter directory, change to Tweets directory
#cd Tweets
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

#sort tweet_list
tweet_list=list(set(tweet_list))
tweet_list.sort()

#import sqlite3
#con=sqlite3.connect('tweets.db')
#DF.to_sql('tweets', con, flavor='sqlite', if_exists='fail', index=True, index_label='index_')
#DF1=pd.io.sql.read_sql('tweets.db', con)
#################################################
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
#################################################
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
#select only rows that reference a particular ticker symbol
DF_tick=DF.ix[tick_index,:]
#unique list of ticker symbols in tweets
tick_list=list(set(tick_list))
#################################################
#count how many occurences of each ticker symbol
tick_count={}
#for x in tick_list:
#    tick_count[x]=0

for x in tick_list:
    i=0
    for j in range(len(DF_tick)):
        if x in DF_tick.ix[j,8]:
            i+=1
    tick_count[x]=i
    
for x in tick_list:
    counter=[1 for j in DF_tick[8] if x in j]
    tick_count[x]=sum(counter)
#################################################
#create a dataframe of ticker counts
DF_tick_count=DataFrame.from_dict(tick_count,orient='index')
#sort ticker counts
DF_tick_count=DF_tick_count.sort(columns=0)
#################################################
#select tweets from a certain period
#see https://docs.python.org/2/library/datetime.html#datetime-objects
#and https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
from datetime import datetime
date_object = datetime.strptime(DF.ix[0,1], '%Y-%m-%d')
date_object1 = datetime.strptime(DF.ix[0,9], '%H:%M:%S')
date_object = date_object.replace(hour=date_object1.hour)
date_object = date_object.replace(minute=date_object1.minute)
date_object = date_object.replace(second=date_object1.second)




