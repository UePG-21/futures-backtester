import pandas as pd

import FuturesBacktester as fb



class BollingerStrategy(fb.Strategy):
    def preprocess(self):
        super(BollingerStrategy, self).preprocess()
        # self.df_price = self.df_price.copy()

        self.df_price["mean"] = self.df_price["close"].rolling(self.window).mean()
        self.df_price["std"] = self.df_price["close"].rolling(self.window).std()
        self.df_price["upper"] = self.df_price["mean"] + self.df_price["std"] * 2
        self.df_price["lower"] = self.df_price["mean"] - self.df_price["std"] * 2

        print(self.df_price)

    def next(self):
        super(BollingerStrategy, self).next()

        close = self.df_price.loc[self.dt, "close"]
        upper = self.df_price.loc[self.dt, "upper"]
        lower = self.df_price.loc[self.dt, "lower"]

        if self.location == 0:
            if close > upper:
                self.broker.buy(label="close")
                self.location = 1
            elif close < lower:
                self.broker.sell(label="close")
                self.location = -1
        elif self.location == -1:
            if close > lower:
                self.broker.close(label="close")
                self.location = 0
        else:
            if close < upper:
                self.broker.close(label="close")
                self.location = 0


if __name__ == "__main__":
    file_path = r"E:\Others\Programming\py_vscode\modules\backtesting\df.csv"
    df = pd.read_csv(file_path, parse_dates=["date"], index_col=["date"])

    account = fb.Account(10000)
    # account.logger.toggle()
    broker = fb.Broker(0.01, 0.00002, 0, 1, account)
    bs = BollingerStrategy("2018-10-10", "2022-02-15", broker)
    bs.set_params(window=15, location=0)
    bs.feed_data(df_price=df.copy())
    bs.preprocess()
    bs.backtest(15, 0)
    bs.analysis()