import oauth2 as oauth
import time

# Set the API endpoint 
url = "http://example.com/photos"

# Set the base oauth_* parameters along with any other parameters required
# for the API call.
params = {
    'oauth_version': "1.0",
    'oauth_nonce': oauth.generate_nonce(),
    'oauth_timestamp': int(time.time())
    'user': 'joestump',
    'photoid': 555555555555
}

# Set up instances of our Token and Consumer. The Consumer.key and 
# Consumer.secret are given to you by the API provider. The Token.key and
# Token.secret is given to you after a three-legged authentication.
token = oauth.Token(key="tok-test-key", secret="tok-test-secret")
consumer = oauth.Consumer(key="con-test-key", secret="con-test-secret")

# Set our token/key parameters
params['oauth_token'] = token.key
params['oauth_consumer_key'] = consumer.key

# Create our request. Change method, etc. accordingly.
req = oauth.Request(method="GET", url=url, parameters=params)

# Sign the request.
signature_method = oauth.SignatureMethod_HMAC_SHA1()
req.sign_request(signature_method, consumer, token)

###########################################################################################################
###########################################################################################################
#see https://developer.linkedin.com/documents/getting-oauth-token-python
#see https://github.com/simplegeo/python-oauth2
#see http://stocktwits.com/developers/docs/authentication#responses
#see http://stocktwits.com/developers/docs/api#oauth-token-docs
import oauth2 as oauth

consumer_key = "e6ea615fc3943c04"
consumer_secret = "cd49fc9a46fa1938c11ca8b45d157aabd7c6e067"
# Create your consumer with the proper key/secret.
consumer = oauth.Consumer(key=consumer_key,secret=consumer_secret)

# Request token URL for Twitter.
request_token_url = "https://api.stocktwits.com/api/2/oauth/token"


# Create our client.
client = oauth.Client(consumer)

# The OAuth Client request works just like httplib2 for the most part.
resp, content = client.request(request_token_url, "POST")
print resp
print content
if resp['status'] != '200':
    raise Exception("Invalid response %s." % resp['status'])
 
request_token = dict(urlparse.parse_qsl(content))
