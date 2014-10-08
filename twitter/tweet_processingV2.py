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
## 11: datetime object of format (year,month,day,hour,minute,second) created below
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
#################################################
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
#################################################
##list of indices corresponding to each ticker symbol
##(try to redo this with map-reduce)
#tick_indices={}
#i=0
#for x in tick_list:
#    i+=1
#    print i
#    print ' out of '
#    print len(tick_list)
#    ind=[j for j in DF_tick.index if x in DF_tick.ix[j,8]]
#    [1 for j in DF_tick[8] if x in j]
#    tick_indices[x]=ind

##save the ticker indices to a csv file in the Tweets directory
#import csv
#writer = csv.writer(open('tick_indices.csv', 'wb'))
#for key, value in tick_indices.items():
#   writer.writerow([key, value])

#print out tweets for AAPL
AAPL_indices=tick_indices['AAPL']
for i in AAPL_indices:
    print DF_tick.ix[i,7]
##################################################################################################
##################################################################################################
##################################################################################################
#manually setup sentiwordnet
cd 
cd /home/sgolbeck/nltk_data/corpora/sentiwordnet
%run sentiwordnet.py
swn_filename = "SentiWordNet_3.0.0_20100705.txt"
swn = SentiWordNetCorpusReader(swn_filename)
swn.senti_synsets('slow')



##################################################################################################
##################################################################################################
##################################################################################################
#create a time window for September
left_window=datetime(2014,9,1,0,0,0)
right_window=datetime(2014,10,1,0,0,0)
#test if date_object is in the window
cond_l=(DF_tick[11]>=left_window)
cond_r=(DF_tick[11]<right_window)
cond=[cond_l[i] and cond_r[i] for i in DF_tick.index]
#select only those rows within the window
DF_sept=DF_tick[cond]
#################################################
#count how many occurences of each ticker symbol
tick_count_sept={}
for x in tick_list:
    counter=[1 for j in DF_sept[8] if x in j]
    tick_count_sept[x]=sum(counter)
#################################################
#create a dataframe of ticker counts
DF_tick_count_sept=DataFrame.from_dict(tick_count_sept,orient='index')
#sort ticker counts
DF_tick_count_sept=DF_tick_count_sept.sort(columns=0,ascending=False)


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


##################################################################################################
##################################################################################################
##################################################################################################
#create a time window for July
left_window=datetime(2014,7,1,0,0,0)
right_window=datetime(2014,8,1,0,0,0)
#test if date_object is in the window
cond_l=(DF_tick[11]>=left_window)
cond_r=(DF_tick[11]<right_window)
cond=[cond_l[i] and cond_r[i] for i in DF_tick.index]
#select only those rows within the window
DF_july=DF_tick[cond]
#################################################
#count how many occurences of each ticker symbol
tick_count_july={}
for x in tick_list:
    counter=[1 for j in DF_july[8] if x in j]
    tick_count_july[x]=sum(counter)
#################################################
#create a dataframe of ticker counts
DF_tick_count_july=DataFrame.from_dict(tick_count_july,orient='index')
#sort ticker counts
DF_tick_count_july=DF_tick_count_july.sort(columns=0,ascending=False)


