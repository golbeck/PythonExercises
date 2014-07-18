import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
from lxml.html import parse
from urllib2 import urlopen
from pandas.io.parsers import TextParser


def _unpack(row,kind='td'):
    elts=row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

def parse_options_data(table):
    rows=table.findall('.//tr')
    header=_unpack(rows[0],kind='th')
    data=[_unpack(r) for r in rows[1:]]
    return TextParser(data,names=header).get_chunk()

########################################################################
########################################################################
parsed=parse(urlopen('http://nymag.com/daily/intelligencer/2013/04/bloombergs-vip-terminal-tweeters.html'))
doc=parsed.getroot()
links=doc.findall('.//a')
links[15:20]
lnk=links[28]
lnk
lnk.get('href')
lnk.text_content()
urls=[lnk.get('href') for lnk in doc.findall('.//a')]
temp=Series(urls[103:205])

for i in range(0,len(temp)):
    temp[i]=temp[i].replace('//www.twitter.com/','')

temp.to_csv("nymag_tweets.csv")



########################################################################
########################################################################
parsed=parse(urlopen('http://www.businessinsider.com/the-best-finance-people-on-twitter-2012-4?op=1'))
doc=parsed.getroot()
links=doc.findall('.//a')
links[15:20]
lnk=links[28]
lnk
lnk.get('href')
lnk.text_content()
urls=[lnk.get('href') for lnk in doc.findall('.//a')]
str_url='https://twitter.com/#!/'

#selects only links that go to twitter feeds
temp=[]
for i in range(0,len(urls)):
    if urls[i] is not None:
        if str_url in urls[i]:
            temp.append(urls[i].replace(str_url,''))

#removes duplicates from the list
temp=Series(list(set(temp)))
temp.to_csv("business_insider.csv")

########################################################################
########################################################################
parsed=parse(urlopen('http://www.linkfest.com/leaderboard'))
doc=parsed.getroot()
links=doc.findall('.//a')
links[15:20]
lnk=links[28]
lnk
lnk.get('href')
lnk.text_content()
urls=[lnk.get('href') for lnk in doc.findall('.//a')]
str_url='http://twitter.com/#!/'

#selects only links that go to twitter feeds
temp=[]
for i in range(0,len(urls)):
    if urls[i] is not None:
        if str_url in urls[i]:
            temp.append(urls[i].replace(str_url,''))

temp=Series(temp)
temp.to_csv("streetEYE.csv")
