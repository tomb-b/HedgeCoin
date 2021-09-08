import numpy as np
from enum import Enum


def proc_data(data, step_size=4):
    gases_prices, gases_limits, gases_used, coins = {}, {}, {}, {}
    i = 0

    for q in query_job:
        if coins.get(q['name']) is None:
            coins[q['name']] = []

        coins[q['name']].append(q)

    for symbol, coin in coins.items():
        while i < len(coin):
            step = step_size

            if i + step_size > len(coin):
                step = len(coin) - i

            gas_used = sum(coin[i: i + step]['gas_used']) / step
            gas_limit = sum(coin[i: i + step]['gas_limit']) / step
            gas_price = sum(coin[i: i + step]['gas_price']) / step

            if gases_prices.get(symbol) is None:
                gases_used[symbol], gases_limits[symbol, gases_prices] = [], [], []

            gases_prices[symbol].append(gas_price)
            gases_used[symbol].append(gas_used)
            gases_limits[symbol].append(gas_limit)
            i += step

    return gases_used, gases_limits, gases_prices


def gen_data(data_min, step_size=4):
    grouped_data = []
    i = 0

    while i < (len(data_min) - 1):
        step = step_size

        if i + step >= len(data_min):
            step = (len(data_min) - 1) - i

        Open = data_min[i][state.Close.value]
        Close = data_min[i + step][state.Close.value]
        Date = data_min[i][state.Date.value]
        Low, High, tradecount, Volume_coin, Volume_usdt = 0, 0, 0, 0, 0

        for data in data_min[i: i + step]:
            tradecount += data[state.Tradecount.value]
            Volume_coin += data[state.Volume_coin.value]
            Volume_usdt += data[state.Volume_usdt.value]

            if data[state.Low.value] < Low:
                Low = data[state.Low.value]
            if data[state.HIgh.value] > High:
                High = data[state.HIgh.value]

        grouped_data.append([Date, Open, High, Low, Close, Volume_coin, Volume_usdt, tradecount])
        i += step

    return grouped_data

class state(Enum):
    Date = 0
    Open = 1
    HIgh = 2
    Low = 3
    Close = 4
    Volume_coin = 5
    Volume_usdt = 6
    Tradecount = 7