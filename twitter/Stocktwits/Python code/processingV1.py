#if in the twitter directory, change to Tweets directory
#################################################
import re
import nltk
import os
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


#generate list of tickers for stocks
pwd_temp=%pwd
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter'
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
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Stocktwits'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Stocktwits'
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

#save the dataframe to a CSV file
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Stocktwits/Processed'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Stocktwits/Processed'
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
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Stocktwits/Processed'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Stocktwits/Processed'
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
##dir1='/home/golbeck/Workspace/PythonExercises/twitter/Stocktwits/Processed'
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
#work computer directory
dir1='/home/sgolbeck/workspace/PythonExercises/twitter/Stocktwits/Processed'
#home computer directory
#dir1='/home/golbeck/Workspace/PythonExercises/twitter/Stocktwits/Processed'
if pwd_temp!=dir1:
    os.chdir(dir1)
writer = csv.writer(open('Stocktwits_database_single_public.csv', 'wb'))
for i in DF_single_public.index:
    writer.writerow([DF_single_public.ix[i,j] for j in DF_single_public.columns])



