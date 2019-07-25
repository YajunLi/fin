import pandas as pd

# 1 triple ewma
def calculate_trix(self, window=15):
    """ Triple Exponential Moving Average Smooth the insignificant movements
            TR(t) / TR(t-1) where
            TR(t) = EMA(EMA(EMA(Price(t)))) over n days period
    """
    self.set_max_win(window)
    # ignore produced warnings for now #TODO
    ema = pd.ewma(self.df['Adj_Close'], span=window, min_periods=window - 1)
    ema = pd.ewma(ema, span=window, min_periods=window - 1)
    ema = pd.ewma(ema, span=window, min_periods=window - 1)

    roc_l = [0]
    for i in range(1, len(self.df.index) - 1):
        roc_l.append((ema[i + 1] - ema[i]) / ema[i])

    name = "trix_%s" % (window)
    self.df[name] = pd.Series(roc_l)
    return name

# 2
def calculate_atr(self, window=14):
     """ Average True Range Shows volatility of market
               ATR(t) = ((n-1) * ATR(t-1) + Tr(t)) / n
             where Tr(t) = Max(Abs(High - Low), Abs(High - Close(t - 1)), Abs(Low - Close(t - 1));
      """
     self.set_max_win(window)
    i = 0
    tr_l = [0]
    for i in range(self.df.index[-1]):
         tr = max(self.df.get_value(i + 1, 'High'),
                 self.df.get_value(i, 'Adj_Close')) - min(self.df.get_value(i + 1, 'Low'),
                                                           self.df.get_value(i, 'Adj_Close'))
           tr_l.append(tr)
    name = 'atr_%s' % (window)
    self.df[name] = pd.ewma(pd.Series(tr_l), span=window, min_periods=window)
    return name

# 3 true strength index
def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    df = df.join(RSI)
    return df

# 4 MSE, RMSE

targets_df = y # pd.read_csv(files[0], header=0, index_col=False)
predict_df = data_cpmax # pd.read_csv(files[1], header=0, index_col=False)

column = targets_df.values

targets = targets_df.as_matrix(columns=[column])

# predict_df[column] = pd.ewma(predict_df[column], com=1, adjust=False)
# predictions = predict_df.as_matrix(columns=[column])
predictions = predict_df[1]
rmse, mse = calc_rmse(predictions, targets)
print("RMSE: %f, MSE: %f" % (rmse, mse))





# plot
import matplotlib.pyplot as plt

plt.plot(cumulative, c='blue')
plt.show()
