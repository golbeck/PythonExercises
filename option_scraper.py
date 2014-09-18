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

def main():
    parsed=parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
    doc=parsed.getroot()
    links=doc.findall('.//a')
    links[15:20]
    lnk=links[28]
    lnk
    lnk.get('href')
    lnk.text_content()
    urls=[lnk.get('href') for lnk in doc.findall('.//a')]
    urls[-10:]
    tables=doc.findall('.//tr')
    calls=tables[9]
    puts=tables[13]
    rows=calls.findall('.//tr')

    _unpack(rows[0],kind='th')
    _unpack(rows[1],kind='td')

    call_data=parse_options_data(calls)
#    put_data=parse_options_data(puts)
    call_data[:10]



if __name__ == '__main__':
    main()
