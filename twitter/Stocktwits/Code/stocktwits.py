###########################################################################################################
###########################################################################################################
#generate token by loading below URL in browser (it is passed in the resulting url)
#https://api.stocktwits.com/api/2/oauth/authorize?client_id=e6ea615fc3943c04&response_type=token&redirect_uri=http://www.stocktwits.com&scope=read,watch_lists,publish_messages,publish_watch_lists,follow_users,follow_stocks
###########################################################################################################
###########################################################################################################
#tools for accessing the API
import oauth2 as oauth
#tweets are output in JSON format. Use this package to load the data into python
import json
#save output to dataframe and then to csv
import pandas as pd

###########################################################################################################
###########################################################################################################
#use for processing the tweets
import nltk
from nltk import word_tokenize

#regular expressions package
import re
#pattern for what the tokens are when using nltk.regexp_tokenize(text,pattern)
#for more info see: http://www.nltk.org/book/ch03.html
pattern = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \$\w+  # currency and percentages, e.g. $12.40, 82%
    | @\S+             # tokenize twitter mentions
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
    '''
###########################################################################################################
###########################################################################################################
def stocktwits_request(max_id,since_id):
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
    if len(max_id)>0:
        #query by max_id
        url=url+'&max='+max_id
    if len(since_id)>0:
        #query by max_id
        url=url+'&since='+since_id
    # OAuth Client request
    try:
        resp, content = client.request(url, "GET")
        response=json.loads(content)
    except:
        print "request failed"
        response={}
        resp={}
        resp['x-ratelimit-remaining']=0
    return {'content':response,'limit':resp['x-ratelimit-remaining']}
###########################################################################################################
###########################################################################################################
def text_processing(text):
    temp_text=text.strip()
    temp_text=re.sub('[,;"\'?():_`/\.]','',temp_text)

    #custom tokenizer using the pattern defined above (beginning of file)
    temp_text=nltk.regexp_tokenize(temp_text, pattern)
    #covnert from unicode to string
    temp_text=[w.encode('utf-8') for w in temp_text]

    #find hashtags and remove #
    temp_text=[re.sub(r'#','',w) for w in temp_text]

    #get rid of links
    temp_text=[w for w in temp_text if not w.startswith('htt')]
    return temp_text
###########################################################################################################
###########################################################################################################
import pytz
from datetime import datetime, timedelta

def is_dst(zonename):
    tz = pytz.timezone(zonename)
    now = pytz.utc.localize(datetime.utcnow())
    return now.astimezone(tz).dst() != timedelta(0)
###########################################################################################################
###########################################################################################################
def adjust_time(temp):
    #New York will stay on EDT until Sunday, November 2, 2014 at 2:00 AM
    #when clocks are moved to Sunday, November 2, 2014 at 01:00:00EST
    #if standard time, offset=5
    #if daylight savings, offset=4
    if is_dst("America/New_York")==True:
        offset=4
    else:
        offset=5

    #twitter times are reported in GMT
    #convert to GMT-5 for EST (NYSE)
    #that is, if you want 8am EST, enter 1pm in the datetime object
    #NYSE opens at 9:30am, closes at 4:00pm

    t_int=int(temp[11:13])-offset
    if t_int<0:
        t_int=24+t_int

    t_int=str(t_int)

    if int(t_int)<10:
        t_int='0'+str(t_int)

    t_string=t_int+temp[13:19]
    return t_string
###########################################################################################################
###########################################################################################################
def adjust_time_date(temp,temp_date):
    #New York will stay on EDT until Sunday, November 2, 2014 at 2:00 AM
    #when clocks are moved to Sunday, November 2, 2014 at 01:00:00EST
    #if standard time, offset=5
    #if daylight savings, offset=4
    if is_dst("America/New_York")==True:
        offset=4
    else:
        offset=5
    
    #temp=messages[i]['created_at'].encode('utf8')

    #twitter times are reported in GMT
    #convert to GMT-5 for EST (NYSE)
    #that is, if you want 8am EST, enter 1pm in the datetime object
    #NYSE opens at 9:30am, closes at 4:00pm

    t_int=int(temp[11:13])-offset
    if t_int<0:
        t_int=24+t_int
        #adjust date backward
        temp_date=temp_date-timedelta(days=1)

    t_int=str(t_int)

    if int(t_int)<10:
        t_int='0'+str(t_int)

    t_string=t_int+temp[13:19]

    temp_month=str(temp_date.month)
    if temp_month<10:
        temp_month='0'+str(temp_month)
    
    temp_day=str(temp_date.day)
    if temp_day<10:
        temp_day='0'+str(temp_day)
    #return date
    temp_=str(temp_date.year)+'-'+temp_month+'-'+temp_day

    return [t_string,temp_]
###########################################################################################################
###########################################################################################################
def adjust_string(string):
    if len(string)<2:
        string='0'+string
    return string
###########################################################################################################
###########################################################################################################
max_id=''
since_id=''
limit=400
M=0
M_temp=999
messages=[]
#download messages as far back as allowed
while limit>M and M_temp>0:
    dict1=stocktwits_request(max_id,since_id)
    response=dict1['content']
    max_id=response['cursor']['max']-1
    max_id=str(max_id)
    since_id=str(response['cursor']['since'])
    message=response['messages']
    #check if query returned an empty list; if yes, then stop
    M_temp=len(message)
    if M_temp>0:
        [messages.append(x) for x in message]
        limit=int(dict1['limit'])
        print limit, max_id, since_id, len(messages)
        print response['messages'][0]['id'], response['messages'][-1]['id']
###########################################################################################################
###########################################################################################################
N=len(messages)
text=[text_processing(messages[i]['body']) for i in range(N)]
created_at=[messages[i]['created_at'] for i in range(N)]
message_id=[messages[i]['id'] for i in range(N)]
symbol=[messages[i]['symbols'][0]['symbol'] for i in range(N)]
classification=[messages[i]['user']['classification'] for i in range(N)]
followers=[messages[i]['user']['followers'] for i in range(N)]
id_=[messages[i]['user']['id'] for i in range(N)]
trading_strategy=[messages[i]['user']['trading_strategy']['approach'] for i in range(N)]
assets_frequently_traded=[messages[i]['user']['trading_strategy']['assets_frequently_traded'] for i in range(N)]
experience=[messages[i]['user']['trading_strategy']['experience'] for i in range(N)]
holding_period=[messages[i]['user']['trading_strategy']['holding_period'] for i in range(N)]
username=[messages[i]['user']['username'] for i in range(N)]

t1=datetime(int(messages[i]['created_at'].encode('utf8')[0:4]),int(messages[i]['created_at'].encode('utf8')[5:7]),int(messages[i]['created_at'].encode('utf8')[8:10]))
#generate date, time of day, day of week
date_=[datetime(int(messages[i]['created_at'].encode('utf8')[0:4]),int(messages[i]['created_at'].encode('utf8')[5:7]),int(messages[i]['created_at'].encode('utf8')[8:10])) for i in range(N)]
time_temp=[adjust_time_date(messages[i]['created_at'].encode('utf8'),date_[i]) for i in range(N)]
time_hour=[time_temp[i][0] for i in range(N)]
time_date=[time_temp[i][1] for i in range(N)]
if is_dst("America/New_York")==True:
    tz_=['EDT' for i in range(N)]
else:
    tz_=['EST' for i in range(N)]



###########################################################################################################
###########################################################################################################
DF_stocktwits=pd.DataFrame()
DF_stocktwits['text']=text
DF_stocktwits['date']=time_date
DF_stocktwits['time']=time_hour
DF_stocktwits['tz']=tz_
DF_stocktwits['message_id']=message_id
DF_stocktwits['symbol']=symbol
DF_stocktwits['classification']=classification
DF_stocktwits['followers']=followers
DF_stocktwits['id']=id_
DF_stocktwits['strategy']=trading_strategy
DF_stocktwits['assets_traded']=assets_frequently_traded
DF_stocktwits['experience']=experience
DF_stocktwits['holding_per']=holding_period
DF_stocktwits['username']=username

###########################################################################################################
###########################################################################################################
current_time=datetime.now()
year=str(current_time.year)
month=adjust_string(str(current_time.month))
day=adjust_string(str(current_time.day))
hr=adjust_string(str(current_time.hour))
mn=adjust_string(str(current_time.minute))
current_time=str(year+'-'+month+'-'+day+'-'+hr+'-'+mn)
str_csv='/home/sgolbeck/Workspace/PythonExercises/twitter/Stocktwits/'+current_time+'PST'+'.csv'
DF_stocktwits.to_csv(str_csv,sep=',',header=True,index=True)

