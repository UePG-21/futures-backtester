"""
@Objective: Futures backtest framework

@Author: UePG

@Version: 4.7
@Date: 05/8/22

@Reference: https://www.backtrader.com/
"""

import copy
import itertools
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd


class Order:
    """
    Order translates the decisions made by a Strategy into a message for the Broker
    """

    def __init__(
        self,
        dt: datetime,
        type: str,
        size: int,
        label: str,
        contract: str = "default",
        **kwargs,
    ):
        self.dt = dt
        self.type = type  # 'buy', 'sell', 'close' or 'hold'
        self.size = size  # number of contracts (>0: long, <0: short, =0:empty)
        self.label = label  # label of the price we quote
        self.contract = contract
        self.quote = None  # quote price
        self.execute = None  # quote * (1 + slippage_ratio)
        self.amount = None  # execute * abs(size)
        self.charge = None  # amount * charge_ratio
        self.margin = None  # amount * margin_ratio
        self.feasible = True  # if this order can be executed
        # additional properties
        for arg_key in kwargs.keys():
            setattr(self, arg_key, kwargs[arg_key])

    def __str__(self):
        if self.type == "hold":
            return f"{self.dt} hold"
        else:
            return f"{self.dt} {self.type} {abs(self.size)} {self.contract} at {self.execute}"


class Account:
    """
    Account, with a built-in Logger, keeps the information of cash, margin and position
    """

    def __init__(self, init_cash: float):
        self.dt: datetime = None
        self.asset = init_cash
        self.cash = init_cash
        self.margin = 0.0
        self.position = 0.0  # curr_margin / ttl_asset
        self.oi = pd.DataFrame()  # open interest detail
        self.logger = Logger()

    def __completed(self):
        self.margin = np.sum(self.oi["margin"])
        self.cash = self.asset - self.margin
        self.position = self.margin / self.asset

    def updated_bar(self, dt: datetime, bar: pd.Series, margin_ratio: float):
        self.dt = dt
        if len(self.oi):
            last_price = copy.deepcopy(self.oi["price"])
            for contract in self.oi.index:
                if contract != "default":
                    self.oi.loc[contract, "price"] = bar[f"{contract}_close"]
                else:
                    self.oi.loc[contract, "price"] = bar["close"]
            self.oi["margin"] = (
                np.abs(self.oi["size"]) * self.oi["price"] * margin_ratio
            )
            self.asset += np.sum(self.oi["size"] * (self.oi["price"] - last_price))
            self.__completed()
        self.logger.log_account(self)

    def updated_order(self, order: Order, margin_ratio: float):
        if order.feasible:
            # continue judging order feasibility for this account
            oi = copy.deepcopy(self.oi)
            asset = copy.deepcopy(self.asset)

            if order.contract not in oi.index:
                order_detail = pd.DataFrame(
                    index=[order.contract],
                    data={
                        "size": order.size,
                        "price": order.execute,
                        "margin": order.margin,
                    },
                )
                oi = pd.concat([oi, order_detail])
                asset -= order.charge
            else:
                c_size = oi.loc[order.contract, "size"]
                c_price = oi.loc[order.contract, "price"]
                if np.sign(order.size) == np.sign(c_size):
                    mean_price = (c_size * c_price + order.size * order.execute) / (
                        c_size + order.size
                    )
                    oi.loc[order.contract, "price"] = mean_price
                    oi.loc[order.contract, "size"] += order.size
                    oi.loc[order.contract, "margin"] += order.margin
                    asset -= order.charge
                else:
                    final_size = c_size + order.size
                    if final_size == 0:
                        oi.drop(order.contract, inplace=True)
                        asset += (order.execute - c_price) * c_size - order.charge
                    elif final_size < 0:
                        if order.size < 0:
                            oi.loc[order.contract, "price"] = order.execute
                            oi.loc[order.contract, "size"] = final_size
                            oi.loc[order.contract, "margin"] = (
                                order.execute * -final_size * margin_ratio
                            )
                            asset += (order.execute - c_price) * c_size - order.charge
                        else:
                            oi.loc[order.contract, "size"] = final_size
                            oi.loc[order.contract, "margin"] = (
                                c_price * -final_size * margin_ratio
                            )
                            asset += (
                                c_price - order.execute
                            ) * order.size - order.charge
                    else:
                        if order.size > 0:
                            oi.loc[order.contract, "price"] = order.execute
                            oi.loc[order.contract, "size"] = final_size
                            oi.loc[order.contract, "margin"] = (
                                order.execute * final_size * margin_ratio
                            )
                            asset += (order.execute - c_price) * c_size - order.charge
                        else:
                            oi.loc[order.contract, "size"] = final_size
                            oi.loc[order.contract, "margin"] = (
                                c_price * final_size * margin_ratio
                            )
                            asset += (
                                c_price - order.execute
                            ) * order.size - order.charge

            if asset >= np.sum(oi["margin"]):  # order is feasible
                # really execute this order and log it
                self.oi = oi
                self.asset = asset
                self.__completed()
                self.logger.log_order(order)


class Logger:
    """
    Logger records the information of an Account and every executed Order
    """

    def __init__(self):
        self.df_account = pd.DataFrame()
        self.df_order = pd.DataFrame()
        self.__mute = False

    def log_account(self, account: Account):
        account_info = {
            "dt": account.dt,
            "asset": account.asset,
            "cash": account.cash,
            "margin": account.margin,
            "position": account.position,
        }
        self.df_account = pd.concat(
            [self.df_account, pd.DataFrame([account_info])], ignore_index=True
        )

    def log_order(self, order: Order):
        self.df_order = pd.concat(
            [self.df_order, pd.DataFrame([order.__dict__])], ignore_index=True
        )

        # print order
        if not self.__mute:
            print(order)

    def toggle(self):
        self.__mute = not self.__mute
        if self.__mute:
            print("# logger mute")
        else:
            print("# logger unmute")


class Broker:
    """
    Broker sets market rules, manages an Account and creates Orders
    """

    def __init__(
        self,
        margin_ratio: float,
        charge_ratio: float,
        slippage_ratio: float,
        multiplier: float,
        account: Account,
    ):
        self.__margin_ratio = margin_ratio
        self.__charge_ratio = charge_ratio
        self.__slippage_ratio = slippage_ratio
        self.__multiplier = multiplier
        self.account = account
        self.dt: datetime = None
        self.bar: pd.Series = None

    def __accept_order(self, order: Order):
        # add a sign to size
        if order.type == "sell":
            order.size = -order.size

        # specify other properties when type is 'close'
        if order.type == "close":
            if order.contract in self.account.oi.index:
                order.size = -self.account.oi.loc[order.contract, "size"]
                if not order.label:
                    if order.size > 0:
                        order.label = "a1"
                    else:
                        order.label = "b1"
            else:
                order.feasible = False

        # set feasibility when type is 'hold'
        if order.type == "hold":
            order.feasible = False

        # calculate other properties
        if order.label:
            # specify label
            if order.contract != "default":
                order.label = f"{order.contract}_{order.label}"
            # calculate prices
            order.quote = self.bar[order.label]
            if not order.quote or np.isnan(
                order.quote
            ):  # no corresponding quotation (0 or np.nan)
                order.feasible = False
            else:
                order.execute = order.quote * (1 + self.__slippage_ratio)
                order.amount = order.execute * abs(order.size)
                order.charge = order.amount * self.__charge_ratio
                order.margin = order.amount * self.__margin_ratio

    def on_bar(self, dt: datetime, bar: pd.Series):
        self.dt = dt
        self.bar = bar
        self.account.updated_bar(dt, bar, self.__margin_ratio)

    def buy(
        self, size: int = 1, contract: str = "default", label: str = "a1", **kwargs
    ):
        order = Order(
            self.dt, "buy", size * self.__multiplier, label, contract, **kwargs
        )
        self.__accept_order(order)
        self.account.updated_order(order, self.__margin_ratio)

    def sell(
        self, size: int = 1, contract: str = "default", label: str = "b1", **kwargs
    ):
        order = Order(
            self.dt, "sell", size * self.__multiplier, label, contract, **kwargs
        )
        self.__accept_order(order)
        self.account.updated_order(order, self.__margin_ratio)

    def close(self, contract: str = "default", label: str = None, **kwargs):
        order = Order(self.dt, "close", None, label, contract, **kwargs)
        self.__accept_order(order)
        self.account.updated_order(order, self.__margin_ratio)

    def hold(self, **kwargs):
        order = Order(self.dt, "hold", 0, None, **kwargs)
        self.__accept_order(order)
        self.account.updated_order(order, self.__margin_ratio)


class Indicator:
    """
    Indicator describes the performance of a Strategy
    """

    def __init__(self, name: str):
        self.__name = name

    def __call__(self, df_account: pd.DataFrame):
        function_map = {
            "acc_asset_net": self.__acc_asset_net,
            "acc_unit_net": self.__acc_unit_net,
            "acc_return": self.__acc_return,
            "ann_return": self.__ann_return,
            "ann_volatility": self.__ann_volatility,
            "max_drawdown": self.__max_drawdown,
            "sharpe": self.__sharpe,
            "calmar": self.__calmar,
            "sortino": self.__sortino,
        }

        return function_map[self.__name](df_account)

    @staticmethod
    def __acc_asset_net(df_account: pd.DataFrame):
        return np.array(df_account.asset)[-1]

    @staticmethod
    def __acc_unit_net(df_account: pd.DataFrame):
        return np.array(df_account.asset)[-1] / np.array(df_account.asset)[0]

    @staticmethod
    def __acc_return(df_account: pd.DataFrame):
        return np.array(df_account.asset)[-1] - np.array(df_account.asset)[0]

    @staticmethod
    def __ann_return(df_account: pd.DataFrame):
        days = math.ceil(
            (np.array(df_account.dt)[-1] - np.array(df_account.dt)[0])
            / np.timedelta64(1, "D")
        )
        return np.log(Indicator("acc_unit_net")(df_account)) * (365 / days)

    @staticmethod
    def __ann_volatility(df_account: pd.DataFrame):
        days = math.ceil(
            (np.array(df_account.dt)[-1] - np.array(df_account.dt)[0])
            / np.timedelta64(1, "D")
        )
        return np.std(np.log(df_account.asset / df_account.asset.shift())) * (
            365 / days
        )

    @staticmethod
    def __max_drawdown(df_account: pd.DataFrame):
        max_acc = np.maximum.accumulate(df_account.asset)
        return ((max_acc - df_account.asset) / max_acc).max()

    @staticmethod
    def __sharpe(df_account: pd.DataFrame, r_f: float = 0.02):
        return (Indicator("ann_return")(df_account) - r_f) / Indicator(
            "ann_volatility"
        )(df_account)

    @staticmethod
    def __calmar(df_account: pd.DataFrame):
        return Indicator("ann_return")(df_account) / Indicator("max_drawdown")(
            df_account
        )

    @staticmethod
    def __sortino(df_account: pd.DataFrame, r_f: float = 0.02):
        days = math.ceil(
            (np.array(df_account.dt)[-1] - np.array(df_account.dt)[0])
            / np.timedelta64(1, "D")
        )
        arr_return_above = np.log(df_account.asset / df_account.asset.shift()) - r_f
        downside_dev = np.sqrt(
            np.sum(arr_return_above[arr_return_above < 0] ** 2)
            / (len(arr_return_above) - 1)
        )
        return (Indicator("ann_return")(df_account) - r_f) / (
            downside_dev * (365 / days)
        )


class Strategy:
    """
    Strategy specifies the trading logic which will be implemented by the Broker
    """

    def __init__(self, start_date: str, end_date: str, broker: Broker):
        self.__itertools = itertools.count(0)  # iterator generator
        self.start_date = start_date
        self.end_date = end_date
        self.broker = broker
        self.iter: int = None
        self.df_price: pd.DataFrame = None
        self.index: pd.DatetimeIndex = None
        self.dt: datetime = None
        self.bar: pd.Series = None
        self.indicators = {
            "acc_asset_net": None,
            "acc_unit_net": None,
            "acc_return": None,
            "ann_return": None,
            "ann_volatility": None,
            "max_drawdown": None,
            "sharpe": None,
            "calmar": None,
            "sortino": None,
        }

    def __on_bar(self):
        self.iter = next(self.__itertools)
        self.dt = self.index[self.iter]
        self.bars = self.df_price.iloc[self.iter]
        self.broker.on_bar(self.dt, self.bar)

    def set_params(self, **kwargs):
        """
        Input parameters to the Strategy, such as window, threshold, etc.
        """
        # should not be overriden
        for arg_key in kwargs.keys():
            setattr(self, arg_key, kwargs[arg_key])

    def feed_data(self, df_price: pd.DataFrame, **kwargs):
        """
        Input data to the Strategy for signal generation and trading simulation
        :param df_price: time series of available prices on bar (ex. close, a1, b1)
        """
        # should not be overriden
        self.df_price = df_price.loc[self.start_date : self.end_date]
        self.index = self.df_price.index  # DatetimeIndex
        for arg_key in kwargs.keys():
            setattr(self, arg_key, kwargs[arg_key])

    def preprocess(self):
        """
        Generate trading signals by using some parameters and data
        """
        # must super() if overriden
        print("Preprocessing...")

    def prenext(self):
        """
        Logic when the Strategy is in buffer and signals can not be calculated
        """
        # must super() if overriden
        self.__on_bar()

    def next(self):
        """
        Logic when the Strategy is mature enough to really execute
        """
        # must be overriden and must super()
        self.__on_bar()

    def stop(self):
        """
        Logic when the Strategy is about to stop as data is near the end
        """
        # must super() if overriden
        self.__on_bar()

    def backtest(self, pre_iter: int, post_iter: int):
        """
        Execute prenext(), next() and stop() in sequence
        :param pre_iter: iter between prenext() and next()
        :param post_iter: iter between next() and stop()
        """
        # must super() if overriden
        print("Backtesting...")

        for _ in range(pre_iter):
            self.prenext()
        for _ in range(pre_iter, len(self.index) - post_iter):
            self.next()
        for _ in range(len(self.index) - post_iter, len(self.index)):
            self.stop()

    def analysis(self, output: str = False, suffix: str = None):
        """
        :param output: absolute path of the result directory
        """
        # must super() if overriden
        # calculate indicators
        print("[Indicators]")
        for name in self.indicators.keys():
            self.indicators[name] = Indicator(name)(
                self.broker.account.logger.df_account
            )
            print(f"{name}: {self.indicators[name]}")

        # output account log and order log
        if output:
            if suffix:
                self.broker.account.logger.df_account.to_csv(
                    output + f"df_account_{suffix}.csv"
                )
                self.broker.account.logger.df_order.to_csv(
                    output + f"df_order_{suffix}.csv"
                )
            else:
                self.broker.account.logger.df_account.to_csv(output + "df_account.csv")
                self.broker.account.logger.df_order.to_csv(output + "df_order.csv")


if __name__ == "__main__":
    # Set path
    dir_main = os.path.abspath(r"..") + "/"

    dir_code = dir_main + "code/"
    dir_config = dir_main + "config/"
    dir_data = dir_main + "data/"
    dir_result = dir_main + "result/"