import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web

def main():
    for ticker in ['AAPL','IBM','MSFT','GOOG']:
        all_data[ticker]=web.get_data_yahoo(ticker,'1/3/200','12/31/2009')
        
    price=DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
    volume=DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})
    returns=(price-price.shift(1))/price
    

if __name__ == '__main__':
    main()
