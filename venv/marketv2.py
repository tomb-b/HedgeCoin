import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from enum import Enum
import requests
import json
from datetime import datetime, timedelta
from process_data import *
import dateutil.parser as dparser


class marketv2(object):
    def __init__(self, device):
        self.currencies, self.dated_prices = {}, {}
        self.sel_currency = ''
        self.minute_iter, self.num_states_per_run, self.num_iter = 0, 200, 0
        self.dollars = random.randint(200, 5000)
        self.coin_history, self.has_coins = [], {}
        self.coins = ('BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'EOS', 'AAVE', 'DOT', 'SOL', 'UNI', 'LINK', 'LTC', 'PAX',
                      'MATIC', 'NEO', 'TRX', 'XLM', 'XRP', 'CELR', 'CVC', 'DASH', 'FIL', 'LRC', 'MKR', 'ONE',
                      'QTUM', 'SC' ,'VET', 'XMR', 'ZEC')
        self.coins_full = ('bitcoin', 'ethereum', 'cardano', 'binance', 'ripple', 'eos', 'polkadot',
                           'solana', 'uniswap', 'chainlink', 'litecoin', 'paxos', 'polygon', 'neo',
                           'tron', 'stellar', 'proton', 'celr', 'civic', 'dash', 'filecoin', 'loopring',
                           'maker', 'harmony', 'qtum', 'siacoin' ,'vechain', 'monero', 'zcash')

        for coin in self.coins:
            self.currencies[coin] = 0.0
        for coin in self.coins:
            data = pd.read_csv(f'data/Binance_{coin}USDT_minute.csv', sep=',',
                               usecols=['date', 'open', 'high', 'low', 'close',
                                        'Volume USDT'], skiprows=1)
            data = data.sort_values('date').values

            if self.dated_prices.get(coin) is None:
                self.dated_prices[coin] = {}
            for i, data_minute in enumerate(data):
                self.dated_prices[coin][f'{data_minute[0]}'] = data_minute[4]

        self.action_space = 200000
        self.observation_space = len(state)
        self.reset()
        self.asset_value = 0

    def reset(self):
        for coin in self.coins:
            if self.currencies[coin] > 0:
                self.has_coins[coin] = coin
            else:
                self.has_coins.pop(coin, None)

        if self.dollars <= 0:
            if len(self.has_coins.values()) > 0:
                rand_coin = random.choice(list(self.has_coins.values()))
            else:
                self.dollars = random.randint(200, 5000)
                rand_coin = random.choice(self.coins)
        else:
            rand_coin = random.choice(self.coins)

        self.sel_currency = rand_coin
        self.data_min = pd.read_csv(f'data/Binance_{rand_coin}USDT_minute.csv', sep=',',
                               usecols=['date', 'open', 'high', 'low', 'close',
                                        'Volume USDT'], skiprows=1)
        self.data_min = self.data_min.sort_values('date').values
        self.data_10m = gen_data(self.data_min, 10)
        #self.data_hour = gen_data(self.data_min, 60)
        #self.data_daily = gen_data(self.data_hour, 24)
        '''self.data_hour = pd.read_csv(f'data/Binance_{rand_coin}USDT_1h.csv', sep=',',
                                usecols=['date', 'open', 'high', 'low', 'close', f'Volume {rand_coin}',
                                         'Volume USDT', 'tradecount'], skiprows=1)
        self.data_daily = pd.read_csv(f'data/Binance_{rand_coin}USDT_d.csv', sep=',',
                                 usecols=['date', 'open', 'high', 'low', 'close', f'Volume {rand_coin}',
                                          'Volume USDT', 'tradecount'], skiprows=1)'''

        #self.data_hour = self.data_hour.sort_values('date')
        #self.data_daily = self.data_daily.sort_values('date')
        #self.data_min_values = self.data_min.values
        #self.minute_iter = random.randint(0, len(self.data_10m) - self.num_states_per_run - 1)
        self.minute_iter = random.randint(0, len(self.data_min) - self.num_states_per_run - 1)
        #self.hourly_iter = int(math.floor(self.minute_iter / 60)) - 1
        #self.daily_iter = int(math.floor(self.minute_iter / 1440)) - 1
        #self.daily_iter, self.hourly_iter, self.minute_iter = 0, 0, 0
        self.set_state()
        self.num_iter = 0
        self.curr_state[state.Dollars.value] = self.dollars
        self.curr_state[state.Coin_Value.value] = self.currencies[self.sel_currency] * \
                                                  self.curr_state[state.Close_minute.value]
        #self.curr_state[state.Asset_value.value] = self.calc_asset_value()
        self.coin_history = []
        #self.dollars = random.randint(1000, 20000)
        #self.curr_hour = datetime.strptime(self.data_min[0][0], '%Y-%m-%d %H:%M:%S').hour
        #self.curr_day = datetime.strptime(self.data_min[0][0], '%Y-%m-%d %H:%M:%S').day

        return self.curr_state

    def set_state(self):
        self.curr_state = [0] * len(state)
        state_iter = 0

        '''for value in self.data_daily[self.daily_iter][1:]:
            self.curr_state[state_iter] = 0 if pd.isnull(value) else value
            state_iter += 1
        for value in self.data_hour[self.hourly_iter][1:]:
            self.curr_state[state_iter] = 0 if pd.isnull(value) else value
            state_iter += 1'''
        for value in self.data_min[self.minute_iter][1:]:
            self.curr_state[state_iter] = 0 if pd.isnull(value) else value
            state_iter += 1

        #self.curr_state[state.Coin_Value.value] = self.currencies[self.sel_currency]
        #self.curr_state[state.Dollars.value] = self.dollars
        return self.curr_state

    def step(self, trade_val_bounded):
        #action[0] = min(1.0, max(-1.0, action))
        #action /= 10
        trade_amount = trade_val_bounded / self.curr_state[state.Close_minute.value]

        if trade_amount >= 0:
            trade_val_bounded = min(self.dollars, trade_val_bounded)
            trade_amount = trade_val_bounded / self.curr_state[state.Close_minute.value]
        else:
            trade_val_bounded = max(-self.currencies[self.sel_currency] * self.curr_state[state.Close_minute.value],
                                    trade_val_bounded)
            trade_amount = max(-self.currencies[self.sel_currency], trade_amount)

        old_currency = self.currencies[self.sel_currency]
        self.currencies[self.sel_currency] += trade_amount
        self.coin_history.append(self.currencies[self.sel_currency])
        self.dollars -= trade_val_bounded
        self.minute_iter += 1
        self.num_iter += 1
        '''next_hour = datetime.strptime(self.data_min[self.minute_iter][0], '%Y-%m-%d %H:%M:%S').hour
        next_day = datetime.strptime(self.data_min[self.minute_iter][0], '%Y-%m-%d %H:%M:%S').day

        if next_hour != self.curr_hour:
            self.hourly_iter += 1
            self.curr_hour = next_hour
        if next_day != self.curr_day:
            self.daily_iter += 1
            self.curr_day = next_day
        if self.minute_iter % 60 == 0 and self.minute_iter > 0:
            self.hourly_iter += 1
        if self.minute_iter % 1440 == 0 and self.minute_iter > 0:
            self.daily_iter += 1'''

        prev_state = self.curr_state
        next_state, done, info = self.set_state(), False, {}
        #next_state[state.Coin_Value.value] = self.currencies[self.sel_currency] * next_state[state.Close_minute.value]
        #next_state[state.Dollars.value] = self.dollars
        reward = trade_val_bounded * (next_state[state.Close_minute.value] - prev_state[state.Close_minute.value])
        #reward = -abs(next_state[state.Close_minute.value] - prev_state[state.Close_minute.value])
        self.asset_value += reward
        self.curr_state[state.Coin_Value.value] = self.currencies[self.sel_currency] * \
                                                  self.curr_state[state.Close_minute.value]
        self.curr_state[state.Dollars.value] = self.dollars
        #date = dparser.parse(self.data_10m[self.minute_iter][0])
        #response = requests.get(f'https://api.coingecko.com/api/v3/coins/{self.sel_currency}/history?date'
         #                       f'={date.day}-{date.month}-{date.year}&localization=false')
        #response = json.loads(response.text)

        if self.minute_iter == len(self.data_min) - 1 or self.num_iter == self.num_states_per_run:
            done = True

        return next_state, reward, done, info

    def render(self):
        prices = []

        for data in self.data_min[self.minute_iter - self.num_states_per_run: self.minute_iter]:
            prices.append(data[4])

        '''plt.ion()
        fig, ax = plt.subplots(2)
        ax[0].plot(range(int(self.num_states_per_run)), prices, label='Prices')
        ax[1].plot(range(int(self.num_states_per_run)), self.coin_history, label='Coins')
        plt.legend()
        plt.draw()
        plt.pause(0.01)
        plt.clf()'''
        value = self.calc_asset_value()
        print('Total value created: ' + str(float(self.asset_value)) + '$')
        print('Total dollar value in pool: ' + str(float(value)) + '$')

    def calc_asset_value(self):
        asset_val = self.dollars

        for coin, num_coins in self.currencies.items():
            date = datetime.strptime(self.data_min[self.minute_iter][0], '%Y-%m-%d %H:%M:%S')
            curr_date = datetime.today()

            if self.dated_prices[coin].get(f'{self.data_min[self.minute_iter][0]}') is None:
                while self.dated_prices[coin].get(f'{date}') is None:
                    date += timedelta(minutes=1)

                    if date > curr_date:
                        while self.dated_prices[coin].get(f'{date}') is None:
                            date -= timedelta(minutes=1)

            asset_val += self.currencies[coin] *  self.dated_prices[coin][f'{date}']

        return asset_val

class state(Enum):
    '''Open_daily = 0
    High_daily = 1
    Low_daily = 2
    Close_daily = 3
    Volume_coin_daily = 4
    Volume_usdt_daily = 5
    tradecount_daily = 6
    Open_hour = 7
    High_hour = 8
    Low_hour = 9
    Close_hour = 10
    Volume_coin_hour = 11
    Volume_usdt_hour = 12
    tradecount_hour = 13'''
    Open_minute = 0
    High_minute = 1
    Low_minute = 2
    Close_minute = 3
    #Volume_coin_min = 4
    Volume_usdt_min = 4
    #tradecount_min = 5
    Coin_Value = 5
    Dollars = 6
    #Asset_value = 9