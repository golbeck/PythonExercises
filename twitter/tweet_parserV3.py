import oauth2 as oauth
import urllib2 as urllib
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import json
import sqlite3

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

def twitterreq(url, method, parameters):
  req = oauth.Request.from_consumer_and_token(oauth_consumer,
                                             token=oauth_token,
                                             http_method=http_method,
                                             http_url=url,
                                             parameters=parameters)

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

def format_date(a,b,c):
    #a: month as string
    #b: day of month as string
    #c: year as string
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    temp=months.index(a)+1
    if temp<10:
        temp='0'+str(temp)
    date_out=c+'-'+str(temp)+'-'+b
    return date_out


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
    return json.load(response)
###############################################    
def tweet_scraper(feed):
    DF0=[]
    count=0
    response=fetchsamples(feed,'')
    count0=len(response)

    for i in range(0,count0):
        temp1=response[i]['created_at'].encode('utf8').split()
        temp_text=response[i]['text'].encode('utf8')
        temp_text=temp_text.replace(',','')
        temp_text=temp_text.replace('\n',' ')
        DF0.append({'id':int(response[i]['id']),'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})

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
                    temp_text=response[i]['text'].encode('utf8')
                    temp_text=temp_text.replace(',','')
                    temp_text=temp_text.replace('\n',' ')
                    DF0.append({'id':int(response[i]['id']),'followers': response[i]['user']['followers_count'],'screen_name':response[i]['user']['screen_name'].encode('utf8'),'text':temp_text,'day_week':temp1[0],'date':format_date(temp1[1],temp1[2],temp1[5]),'time':temp1[3],'retweet_count':response[i]['retweet_count'],'user_id':response[i]['user']['id']})
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
feed='CNBC'
tweet_scraper(feed)

temp1=pd.io.parsers.read_table('nymag_tweets.csv',sep=',',index_col=0,header=None)
temp2=pd.io.parsers.read_table('streetEYE.csv',sep=',',index_col=0,header=None)
temp3=pd.io.parsers.read_table('business_insider.csv',sep=',',index_col=0,header=None)
temp=temp1.merge(temp2,how='outer')
temp=temp.merge(temp3,how='outer')
temp=temp.drop([102])

temp.to_csv('tweet_master_list.csv',header=False)


