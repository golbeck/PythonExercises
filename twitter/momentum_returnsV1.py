
left_window=datetime(2014,9,1)
right_window=datetime(2014,10,1)
import pandas.io.data

df = pd.io.data.get_data_yahoo('AAPL',start=left_window,end=right_window)
close_px = df['Adj Close']
rets = close_px / close_px.shift(1) - 1
rets.head()
close_px.pct_change().head()

