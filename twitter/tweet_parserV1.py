import oauth2 as oauth
import urllib2 as urllib
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import json

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




def fetchsamples():
  url = "https://api.twitter.com/1.1/search/tweets.json?q=microsoft"
  parameters = []
  response = twitterreq(url, "GET", parameters)
  return json.load(response)
  for line in response:
    print line.strip()

if __name__ == '__main__':
  fetchsamples()

#def fetchsamples():
#  url = "https://api.twitter.com/1.1/search/tweets.json?q=microsoft"
#  parameters = []
#  response = twitterreq(url, "GET", parameters)
#  return json.load(response)
## for line in response:
## print line.strip()

##if __name__ == '__main__':
## fetchsamples()

myResults = fetchsamples()
#print type(myResults)
#print myResults.keys()
#print myResults["statuses"]
#print type(myResults["statuses"])
results = myResults["statuses"]
#print results[0]
#print type(results[0])
#print results[0].keys()
#print results[0]["text"]
#print results[2]["text"]
#print results[5]["text"]
#for i in range(10):
#	print results[i]["text"]
	
	
###############################################
#build dictionary
afinnfile = open("AFINN-111.txt")
scores = {}
for line in afinnfile:
    term, score  = line.split("\t") 
    scores[term] = float(score)
#print scores.items()

###############################################
#read in tweets and save into a dictionary
atweetfile = open("output.txt")
tweets = []
for line in atweetfile:
    try:
        tweets.append(json.loads(line))
    except:
        pass
            
print len(tweets)
tweet = tweets[0]
print type(tweet)
print tweet.keys()
print type(tweet["text"])
print tweet["text"]
