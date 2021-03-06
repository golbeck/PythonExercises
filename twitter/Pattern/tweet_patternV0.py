import numpy as np
from pandas import DataFrame, Series
import pandas as pd
#use for processing the tweets
import nltk
from nltk import word_tokenize
#regular expressions package
import re
##############################################################################################
##############################################################################################
##############################################################################################
cd /home/sgolbeck/workspace/PythonExercises/twitter
#generate list of tickers for stocks
ticklist1=pd.io.parsers.read_table('Ticker list/AMEX.csv',sep=',')
ticklist2=pd.io.parsers.read_table('Ticker list/NYSE.csv',sep=',')
ticklist3=pd.io.parsers.read_table('Ticker list/NASDAQ.csv',sep=',')
ticklist=ticklist1.merge(ticklist2,how='outer')
ticklist=ticklist.merge(ticklist3,how='outer')
tickers1=ticklist[ticklist['MarketCap']>0.0]
tickers2=list(tickers1['Symbol'])
tickers3=[re.sub(r'\s','',w) for w in tickers2]
tickers=list(set(tickers3))
##############################################################################################
##############################################################################################
##############################################################################################
from pattern.web import Twitter
#number of tickers to search
N=len(tickers)
#number of tweets to download
M=2000
#Dataframe
DF0=[]
#loop
t = Twitter()
for j in range(N):
    tick='$'+tickers[j]
    i = None
    for tweet in t.search(tick, start=i, count=M):
#        temp_text=re.sub('[,;"\'?():_`/\.]','',tweet.text)
#        temp_text=temp_text.strip()
        temp_text=tweet.text.strip()
        temp_text.replace('\n',' ')
        DF0.append({'id':tweet.id,'tickers':tick,'screen_name':tweet.author,'text':temp_text,'time':tweet.date})
#        print tweet.text
        i = tweet.id

DF2=DF0

for i in range(len(DF2)):
#    DF2[i]['text']=DF2[i]['text'].encode('utf-8')
#    DF2[i]['text'].encode('utf-8')
    DF2[i]['text'].replace(r"\\",'')
    DF2[i]['text'].replace('\n','')
    DF2[i]['text'].strip()
    DF1=DataFrame(DF2[0:i])
    DF1.to_csv('DF1.csv',sep=',',header=False,index=True)

DF1=DataFrame(DF0)
import datetime
clk=datetime.datetime.now().strftime('%Y_%m_%d')
str_csv='Tweets_'+clk+'.csv'
DF1.to_csv(str_csv,sep=',',header=False,index=True)

for i in range(len(DF1)):
    print DF1.ix[i,'text'].encode('utf-8')
##############################################################################################
##############################################################################################
##############################################################################################
def format_date(a,b,c):
    #converts date to standard format
    #a: month as string
    #b: day of month as string
    #c: year as string
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    temp=months.index(a)+1
    if temp<10:
        temp='0'+str(temp)
    date_out=c+'-'+str(temp)+'-'+b
    return date_out
##############################################################################################
##############################################################################################
##############################################################################################
def fetchsamples(feed,max_id):
    #to build a query, see:
    #https://dev.twitter.com/docs/using-search
    # feed: string containing the term to be searched for (see above link)
    # max_id: string with the ID of the most recent tweet to include in the results
#    url = "https://api.twitter.com/1.1/search/tweets.json?q="
    url="https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name="
    url=url+feed
    #download the maximum number of tweets
    url=url+'&count=200'
    url=url+'&exclude_replies=true'
    url=url+'&include_rts=false'
    if len(max_id)>0:
        url=url+'&max_id='+max_id
    parameters = []
    response = twitterreq(url, "GET", parameters)
    #convert a json object to a python object (in this case, returns a list)
    try:
        response=json.load(response)
    except:
        response=[]
    return response
##############################################################################################
##############################################################################################
##############################################################################################
def feed_create_list(i,response,pattern,tickers,DF0):
    temp1=response[i]['created_at'].encode('utf8').split()
    temp_text=response[i]['text'].strip()
    temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)

    #custom tokenizer using the pattern defined above (beginning of file)
    temp_text=nltk.regexp_tokenize(temp_text, pattern)
    #covnert from unicode to string
    temp_text=[w.encode('utf-8') for w in temp_text]

    #find hashtags and remove #
    temp_text=[re.sub(r'#','',w) for w in temp_text]

    #get rid of links
    temp_text=[w for w in temp_text if not w.startswith('htt')]

    #find ticker symbols
    ticks=[w for w in temp_text if w.startswith('$')]
    if len(ticks)>0.0:
        ticks=[w[1:] for w in ticks if w[1:] in tickers]

    DF0.append({'id':int(response[i]['id']),'tickers':ticks,'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})

    return DF0
##############################################################################################
##############################################################################################
##############################################################################################
tweets1=pd.io.parsers.read_table('nymag_tweets.csv',sep=',',index_col=0,header=None)
tweets2=pd.io.parsers.read_table('streetEYE.csv',sep=',',index_col=0,header=None)
tweets3=pd.io.parsers.read_table('business_insider.csv',sep=',',index_col=0,header=None)
tweets=tweets1.merge(tweets2,how='outer')
tweets=tweets.merge(tweets3,how='outer')
tweets=tweets.drop([102])
tweets=tweets.reindex(range(0,len(tweets)))
tweets.to_csv('tweet_master_list.csv',header=False)
#remove bad names
tweets=tweets[tweets[1]!='trovwolv']
tweets=tweets[tweets[1]!='ldelevigne']
tweets=tweets[tweets[1]!='zlehn']
tweets.index=range(len(tweets))
tweets=tweets.drop([98],axis=0)
tweets=tweets[tweets[1]!='pwmorski']
tweets=tweets[tweets[1]!='dutch_book']
tweets=tweets[tweets[1]!='pragcapitalist']
tweets=tweets[tweets[1]!='trendrida']
tweets=tweets[tweets[1]!='saraeisenfx']
tweets=tweets[tweets[1]!='efficiencynow']
tweets=tweets[tweets[1]!='austan_goolsbee']
tweets.index=range(len(tweets))
for i in range(0,len(tweets)):
    tweets.ix[i,1]=tweets.ix[i,1].lower()
tweets=list(set(tweets[1]))
tweets.sort()

#for j in range(0,len(temp)):
count_API=0
for j in range(228,len(tweets)):
    feed=tweets[j]
    DF0=[]
    count=0
    #set count0 to 1 so that the nested if statement is initially executed
    max_id=''

    N=3000

    while count <= N:
        #load the output from the twitter API into response
        try:
            response=fetchsamples(feed,max_id)
            count0=len(response)
        except:
            count0=0
        count_API+=1
        print 'API calls: '+str(count_API)
        
        if count0>0:
            for i in range(0,count0):
                DF0=feed_create_list(i,response,pattern,tickers,DF0)

            max_id=str(int(response[count0-1]['id'])-1)
#            print max_id
            count+=count0
            print count
        else:
            count=N+1
            print count

    DF1=DataFrame(DF0)
    str_csv='Tweets/'+feed+'.csv'
    DF1.to_csv(str_csv,sep=',',header=False,index=True)



