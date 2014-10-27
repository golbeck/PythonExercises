###########################################################################################################
###########################################################################################################
#generate token (it is passed in the resulting url)
https://api.stocktwits.com/api/2/oauth/authorize?client_id=e6ea615fc3943c04&response_type=token&redirect_uri=http://www.stocktwits.com&scope=read,watch_lists,publish_messages,publish_watch_lists,follow_users,follow_stocks
###########################################################################################################
###########################################################################################################
#see https://developer.linkedin.com/documents/getting-oauth-token-python
#see https://github.com/simplegeo/python-oauth2
#see http://stocktwits.com/developers/docs/authentication#responses
#see http://stocktwits.com/developers/docs/api#oauth-token-docs
#tools for accessing the API
import oauth2 as oauth
<<<<<<< HEAD
import urllib2 as urllib
#tweets are output in JSON format. Use this package to load the data into python
import json
=======
import urlparse 

curl -X POST https://api.stocktwits.com/api/2/oauth/token -d 'client_id=e6ea615fc3943c04&client_secret=cd49fc9a46fa1938c11ca8b45d157aabd7c6e067&code=<code>&grant_type=authorization_code&redirect_uri=http://www.example.com'

curl -X GET https://api.stocktwits.com/api/2/oauth/authorize -d 'client_id=e6ea615fc3943c04&response_type=code&redirect_uri=http://www.stocktwits.com&scope=read,watch_lists,publish_messages,publish_watch_lists,follow_users,follow_stocks'
>>>>>>> 108031d34579cf5065218ff827933907f4c28446

consumer_key = "e6ea615fc3943c04"
consumer_secret = "cd49fc9a46fa1938c11ca8b45d157aabd7c6e067"
access_token='7d12b1c80b938756360078a3cfbd1651acd8b302'
# Create your consumer with the proper key/secret.
<<<<<<< HEAD
oauth_consumer = oauth.Consumer(key=consumer_key,secret=consumer_secret)

=======
consumer = oauth.Consumer(key=consumer_key,secret=consumer_secret)

# Request token URL for Twitter.
#request_token_url = "https://api.stocktwits.com/api/2/oauth/token"
request_token_url = "https://api.stocktwits.com/api/2/oauth/authorize"
>>>>>>> 108031d34579cf5065218ff827933907f4c28446

# Create our client.
client = oauth.Client(oauth_consumer)
url='https://api.stocktwits.com/api/2/streams/suggested.json?access_token='
#download the maximum number of tweets
url=url+access_token
# The OAuth Client request works just like httplib2 for the most part.
resp, content = client.request(url, "GET")
print resp
print content
if resp['status'] != '200':
    raise Exception("Invalid response %s." % resp['status'])

response=json.loads(content)
keys_=response.keys()
dict_cursor=response['cursor']
max_id=dict_cursor['max']-1
min_id=dict_cursor['since']
messages=response['messages']
###########################################################################################################
###########################################################################################################
def stocktwits_request(max_id):
    #max_id: id larger than any message you want to obtain (enter '' if you want the latest messages)

    consumer_key = "e6ea615fc3943c04"
    consumer_secret = "cd49fc9a46fa1938c11ca8b45d157aabd7c6e067"
    access_token='7d12b1c80b938756360078a3cfbd1651acd8b302'
    # Create your consumer with the proper key/secret.
    oauth_consumer = oauth.Consumer(key=consumer_key,secret=consumer_secret)

    # Create our client.
    client = oauth.Client(oauth_consumer)
    url='https://api.stocktwits.com/api/2/streams/suggested.json?access_token='
    #download the maximum number of tweets
    url=url+access_token
    url=url+'&max='+max_id
    # The OAuth Client request works just like httplib2 for the most part.
    try:
        resp, content = client.request(url, "GET")
        response=json.loads(content)
    except:
        print "request failed"
        response={}
        resp={}
        resp['x-ratelimit-remaining']=0
    return {'content':response,'limit':resp['x-ratelimit-remaining']}


limit=400
M=400
message=[]
max_id=''
while limit>M:
    dict1=stocktwits_request(max_id)
    response=dict1['content']
    max_id=response['cursor']['max']-1
    max_id=str(max_id)
    message=response['messages']
    [messages.append(x) for x in message]
    limit=int(dict1['limit'])
    print limit



