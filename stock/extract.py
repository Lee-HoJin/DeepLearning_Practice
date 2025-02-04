from pykrx import stock
import ta
import pandas as pd
import ta.momentum


df = stock.get_market_ohlcv_by_date("20200101", "20250101", "247540")
print(df.head())

df['RSI'] = ta.momentum.rsi(df['종가'], window = 14)
print(df[['종가', 'RSI']].tail())

df_trading = stock.get_market_trading_value_by_date("20200101", "20250101", "247540")
print(df_trading)

df_close = df['종가']
df = df.drop(columns=['종가'])

df_trading.drop("전체", axis = 1, inplace = True)

df_combined = pd.concat([df, df_trading, df_close], axis = 1)
print(df_combined.head())

df_combined.to_csv("./에코프로비엠.csv", encoding="utf-8-sig")  # utf-8-sig는 한글 깨짐 방지용
