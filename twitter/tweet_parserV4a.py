#tools for accessing the twitter API
import oauth2 as oauth
import urllib2 as urllib
#tweets are output in JSON format. Use this package to load the data into python
import json
#use to build a database of tweets
import sqlite3
#use for processing the tweets
import nltk
from nltk import word_tokenize

#edit stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

#lemmatizer: Returns the input word unchanged if it cannot be found in WordNet.
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

#regular expressions package
import re
#pattern for what the tokens are when using nltk.regexp_tokenize(text,pattern)
#for more info see: http://www.nltk.org/book/ch03.html
pattern = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
    '''

import numpy as np
from pandas import DataFrame, Series
import pandas as pd

#generate list of tickers for stocks
temp1=pd.io.parsers.read_table('Ticker list/AMEX.csv',sep=',')
temp2=pd.io.parsers.read_table('Ticker list/NYSE.csv',sep=',')
temp3=pd.io.parsers.read_table('Ticker list/NASDAQ.csv',sep=',')
temp=temp1.merge(temp2,how='outer')
temp=temp.merge(temp3,how='outer')
tickers=list(temp['Symbol'])
tickers=[w.lower() for w in tickers]

###############################################    
###############################################    
###############################################    

# See Assignment 1 instructions or README for how to get these credentials
access_token_key = "89257335-W8LCjQPcTMIpJX9vx41Niqe5ecMtw0tf2m65qsuVn"
access_token_secret = "5tmU9RDxP3tiFShmtDcFE5VVzWy7dGBRvvDp6uwoZWyW2"

consumer_key = "qknqCAZAOOcpejiYkyYZ00VZr"
consumer_secret = "xQM8ynjjXQxy6jWus4qTlCDEPItjZyxqhnAEbbmmUj2Q1JlX5w"

_debug = 0

oauth_token = oauth.Token(key=access_token_key, secret=access_token_secret)
oauth_consumer = oauth.Consumer(key=consumer_key, secret=consumer_secret)

signature_method_hmac_sha1 = oauth.SignatureMethod_HMAC_SHA1()

http_method = "GET"


http_handler = urllib.HTTPHandler(debuglevel=_debug)
https_handler = urllib.HTTPSHandler(debuglevel=_debug)

'''
Construct, sign, and open a twitter request
using the hard-coded credentials above.
'''
###############################################    
###############################################    
###############################################    

def twitterreq(url, method, parameters):
    req = oauth.Request.from_consumer_and_token(oauth_consumer,token=oauth_token,
            http_method=http_method,http_url=url,parameters=parameters)

    req.sign_request(signature_method_hmac_sha1, oauth_consumer, oauth_token)

    headers = req.to_header()

    if http_method == "POST":
        encoded_post_data = req.to_postdata()
    else:
        encoded_post_data = None
        url = req.to_url()

    opener = urllib.OpenerDirector()
    opener.add_handler(http_handler)
    opener.add_handler(https_handler)

    response = opener.open(url, encoded_post_data)

    return response


###############################################    
###############################################    
###############################################    

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


###############################################    
###############################################    
###############################################    
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
    return json.load(response)
###############################################    
###############################################    
###############################################    
def find_ticker(temp_text,tickers):
    ticks=[w for w in temp_text if w in tickers] 
    return ticks
###############################################    
###############################################    
###############################################    
def tweet_scraper(feed):
    DF0=[]
    count=0
    #load the output from the twitter API into response
    response=fetchsamples(feed,'')
    count0=len(response)

    for i in range(count0):
        temp1=response[i]['created_at'].encode('utf-8').split()
        #first strip of endline characters
        temp_text=response[i]['text'].strip()
        temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)
#        #prefix regular expressions with the letter r (meaning "raw"), which instructs 
#        #the Python interpreter to treat the string literally, rather than processing any backslashed characters it contains
#        temp_text=re.split(r'\W+',temp_text)

        #custom tokenizer using the pattern defined above (beginning of file)
        temp_text=nltk.regexp_tokenize(temp_text, pattern)
        #covnert from unicode to string
        temp_text=[w.encode('utf-8') for w in temp_text]
        #lower case
        temp_text=[w.lower() for w in temp_text]
        
#        #lemmatize the words
#        temp_text=[wnl.lemmatize(w).encode('utf-8') for w in temp_text if w.isalpha()]
        #save $+words as labels for the ticker first before removing them

        #first process by finding words with @, $ or # at
        temp_text=[re.sub(r'@','',w) for w in temp_text]
        temp_text=[re.sub(r'$','',w) for w in temp_text]
        temp_text=[re.sub(r'#','',w) for w in temp_text]
#        #then remove punctuation (and non-alphanumeric characters)
#        temp_text=[w for w in temp_text if w.isalpha()]

        #get rid of links
        temp_text=[w for w in temp_text if not w.startswith('htt')]

        #find ticker symbols
        ticks=[w for w in temp_text if w in tickers]

        DF0.append({'id':int(response[i]['id']),'tickers':ticks,'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode ('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})


    max_id=str(int(response[count0-1]['id'])-1)
    print max_id
    count+=count0
    print count


    while count <= 3200:
        if count0>0:
            response=fetchsamples(feed,max_id)
            count0=len(response)
            if count0>0:
                for i in range(0,count0):
                    temp1=response[i]['created_at'].encode('utf8').split()
                    temp_text=response[i]['text'].strip()
                    temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)

                    #custom tokenizer using the pattern defined above (beginning of file)
                    temp_text=nltk.regexp_tokenize(temp_text, pattern)
                    #covnert from unicode to string
                    temp_text=[w.encode('utf-8') for w in temp_text]
                    #lower case
                    temp_text=[w.lower() for w in temp_text]

                    #first process by finding words with $ or ! at
                    temp_text=[re.sub(r'@','',w) for w in temp_text]
                    temp_text=[re.sub(r'$','',w) for w in temp_text]
                    temp_text=[re.sub(r'#','',w) for w in temp_text]

                    #get rid of links
                    temp_text=[w for w in temp_text if not w.startswith('htt')]

                    #find ticker symbols
                    ticks=[w for w in temp_text if w in tickers]

                    DF0.append({'id':int(response[i]['id']),'tickers':ticks,'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})
                    max_id=str(int(response[count0-1]['id'])-1)
                print max_id
                count+=count0
                print count
            else:
                count=3201

    DF1=DataFrame(DF0)
    str_csv=feed+'.csv'
    DF1.to_csv(str_csv,sep=',',header=False,index=True)
    return 0.0
###############################################    
###############################################    
###############################################
#feed='CNBC'
#tweet_scraper(feed)

temp1=pd.io.parsers.read_table('nymag_tweets.csv',sep=',',index_col=0,header=None)
temp2=pd.io.parsers.read_table('streetEYE.csv',sep=',',index_col=0,header=None)
temp3=pd.io.parsers.read_table('business_insider.csv',sep=',',index_col=0,header=None)
temp=temp1.merge(temp2,how='outer')
temp=temp.merge(temp3,how='outer')
temp=temp.drop([102])
temp=temp.reindex(range(0,len(temp)))
temp.to_csv('tweet_master_list.csv',header=False)

#for i in range(0,len(temp)):
for i in range(0,1):
    feed=temp.ix[i,1]
#    tweet_scraper(feed)
    DF0=[]
    count=0
    #load the output from the twitter API into response
    response=fetchsamples(feed,'')
    count0=len(response)

    for i in range(count0):
        temp1=response[i]['created_at'].encode('utf-8').split()
        #first strip of endline characters
        temp_text=response[i]['text'].strip()
        temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)
#        #prefix regular expressions with the letter r (meaning "raw"), which instructs 
#        #the Python interpreter to treat the string literally, rather than processing any backslashed characters it contains
#        temp_text=re.split(r'\W+',temp_text)

        #custom tokenizer using the pattern defined above (beginning of file)
        temp_text=nltk.regexp_tokenize(temp_text, pattern)
        #covnert from unicode to string
        temp_text=[w.encode('utf-8') for w in temp_text]
        #lower case
        temp_text=[w.lower() for w in temp_text]
        
#        #lemmatize the words
#        temp_text=[wnl.lemmatize(w).encode('utf-8') for w in temp_text if w.isalpha()]
        #save $+words as labels for the ticker first before removing them

        #first process by finding words with @, $ or # at
        temp_text=[re.sub(r'@','',w) for w in temp_text]
        temp_text=[re.sub(r'$','',w) for w in temp_text]
        temp_text=[re.sub(r'#','',w) for w in temp_text]
#        #then remove punctuation (and non-alphanumeric characters)
#        temp_text=[w for w in temp_text if w.isalpha()]

        #get rid of links
        temp_text=[w for w in temp_text if not w.startswith('htt')]

        #find ticker symbols
        ticks=[w for w in temp_text if w in tickers]

        DF0.append({'id':int(response[i]['id']),'tickers':ticks,'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode ('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})


    max_id=str(int(response[count0-1]['id'])-1)
    print max_id
    count+=count0
    print count


    while count <= 3200:
        if count0>0:
            response=fetchsamples(feed,max_id)
            count0=len(response)
            if count0>0:
                for i in range(0,count0):
                    temp1=response[i]['created_at'].encode('utf8').split()
                    temp_text=response[i]['text'].strip()
                    temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)

                    #custom tokenizer using the pattern defined above (beginning of file)
                    temp_text=nltk.regexp_tokenize(temp_text, pattern)
                    #covnert from unicode to string
                    temp_text=[w.encode('utf-8') for w in temp_text]
                    #lower case
                    temp_text=[w.lower() for w in temp_text]

                    #first process by finding words with $ or ! at
                    temp_text=[re.sub(r'@','',w) for w in temp_text]
                    temp_text=[re.sub(r'$','',w) for w in temp_text]
                    temp_text=[re.sub(r'#','',w) for w in temp_text]

                    #get rid of links
                    temp_text=[w for w in temp_text if not w.startswith('htt')]

                    #find ticker symbols
                    ticks=[w for w in temp_text if w in tickers]

                    DF0.append({'id':int(response[i]['id']),'tickers':ticks,'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})
                    max_id=str(int(response[count0-1]['id'])-1)
                print max_id
                count+=count0
                print count
            else:
                count=3201

    DF1=DataFrame(DF0)
    str_csv=feed+'.csv'
    DF1.to_csv(str_csv,sep=',',header=False,index=True)



