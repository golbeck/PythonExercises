#dependency: tweet_processingV3.py and tweet_sentimentV1.py
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
#################################################

#only examine tweets that reference a single stock ticker
tick_indices_unique=[]
for i in DF_tick.index:
    if len(DF_tick.ix[i,8])==1:
        tick_indices_unique.append(i)


#DataFrame with only tweets that reference a single ticker
DF_unique=DF_tick.ix[tick_indices_unique]

#intersection of august tweets with tickers & tweets that reference a single ticker
tick_indices_unique_aug=[i for i in DF_aug.index if i in tick_indices_unique]

#determine summary statistics for pos and neg sentiment for each ticker
#restrict dataframe to only tickers with at least m tweets
m=20
DF_tick_count_aug_m=DF_tick_count_aug[DF_tick_count_aug[0]>m]

#numpy for computing statistics
import numpy as np
#dictionaries for saving summary statistics for each ticker
sent_pos_aug={}
sent_neg_aug={}
#loop over tickers
for x in DF_tick_count_aug_m.index:
    #empty lists to hold the sentiment scores
    sent_pos=[]
    sent_neg=[]
    #generate indices corresponding to tweets mentioning a single ticker in August
    x_indices=[j for j in tick_indices_unique_aug if j in tick_indices[x]]
    #loop over each tweet, saving the positive and negative sentiment
    for i in x_indices:
        sent_pos.append(DF_aug.ix[i,12][0])
        sent_neg.append(DF_aug.ix[i,12][1])
    #save the summary statistics for the ticker in the dictionaries
    sent_pos_aug[x]=[np.mean(sent_pos),np.median(sent_pos),np.std(sent_pos),len(sent_pos)]
    sent_neg_aug[x]=[np.mean(sent_neg),np.median(sent_neg),np.std(sent_neg),len(sent_neg)]
#convert dictionaries to DataFrame
DF_sent_pos_aug=DataFrame.from_dict(sent_pos_aug,orient='index')
DF_sent_neg_aug=DataFrame.from_dict(sent_neg_aug,orient='index')
#generate sorted lists, ranking tickers according to both positive and negative sentiment
DF_sent_pos_aug=DF_sent_pos_aug.sort(columns=0,ascending=False)
DF_sent_neg_aug=DF_sent_neg_aug.sort(columns=0,ascending=False)





