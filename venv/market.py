import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from enum import Enum
from google.cloud import bigquery
from google.oauth2 import service_account
import process_data
#from lstmModel import *


class market(object):
    def __init__(self, device, step_size=4):
        self.currencies, self.currency_data, self.assets_daily, faulty_currency = {}, {}, {}, {}
        self.iter, self.asset_value, self.initial_asset_value, self.sel_currency = 0, 0.0, 0.0, ''
        self.dollars = random.randint(0, 10000)
        self.coins = ('BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'EOS', 'AAVE', 'DOT', 'SOL', 'UNI', 'LINK', 'LTC',
                      'MATIC', 'NEO', 'TRX', 'XLM', 'XRP')
        credentials = service_account.Credentials.from_service_account_file(
            '/Users/tombjurenlind/PycharmProjects/HedgeCoin/venv/data/data-key.json')
        client = bigquery.Client(credentials=credentials)
        query = """select * from `bigquery-public-data.crypto_ethereum.transactions` limit 1000"""
        #model = LSTM(len(state), 1).to(device)
        #torch.load(model.state_dict(), "lstmPricePredict")

        '''data = pd.read_csv('data/all_currencies.csv', sep=',',
                           usecols=['Date','Symbol', 'Open', 'High', 'Low',
                                    'Close', 'Volume', 'Market Cap'])'''
        rand_coin = random.choice(self.coins)
        data_min = pd.read_csv(f'data/Binance_{rand_coin}_minute.csv.csv', sep=',',
                           usecols=['unix' ,'date', 'symbol', 'open', 'high', 'low', 'close', f'Volume_{rand_coin}',
                                    'Volume_USDT', 'tradecount'], skiprows=1)
        data_hour = pd.read_csv(f'data/Binance_{rand_coin}_1h.csv.csv', sep=',',
                           usecols=['unix' ,'date', 'symbol', 'open', 'high', 'low', 'close', f'Volume_{rand_coin}',
                                    'Volume_USDT', 'tradecount'], skiprows=1)
        data_daily = pd.read_csv(f'data/Binance_{rand_coin}_d.csv.csv', sep=',',
                           usecols=['unix' ,'date', 'symbol', 'open', 'high', 'low', 'close', f'Volume_{rand_coin}',
                                    'Volume_USDT', 'tradecount'], skiprows=1)
        query = f"""select tx.token_address, t.name, tx.from_address, tx.to_address, tx.value, t.total_supply,
        tx.block_timestamp, trans.gas as gas_limit, trans.gas_price, trans.receipt_gas_used as gas_used
        from 
        `bigquery-public-data.crypto_ethereum.token_transfers` tx 
        JOIN `bigquery-public-data.crypto_ethereum.tokens` t on tx.token_address = t.address
        JOIN `bigquery-public-data.crypto_ethereum.transactions` trans on trans.`hash` = tx.transaction_hash 
        where t.name IN {self.coins}
        order by date(tx.block_timestamp) desc
        LIMIT 50000"""
        query_job = client.query(query)
        gas_used, gas_limits, gas_prices = process_data.process_query(query_job, step_size)

        data = data.values
        data.sort(key=lambda data: data[0])
        start_date = min(data[:, 0])

        for Date, Symbol, Open, High, Low, Close, Volume, Market_Cap in data:
            if pd.isnull(Market_Cap) or pd.isnull(Volume) or pd.isnull(Open) or pd.isnull(Close)\
                    or pd.isnull(High) or pd.isnull(Low) or pd.isnull(Date) or pd.isnull(Symbol) or Market_Cap == 0:
                faulty_currency[Symbol] = True
                continue
            if self.assets_daily.get(Date) is None:
                self.assets_daily[Date] = {}
            if self.currency_data.get(Symbol) is None:
                self.currency_data[Symbol] = []
                #self.asset_value += currency * Close

        for Date, Symbol, Open, High, Low, Close, Volume, Market_Cap in data.values:
            if pd.isnull(Market_Cap) or pd.isnull(Volume) or pd.isnull(Open) or pd.isnull(Close) \
                    or pd.isnull(High) or pd.isnull(Low) or pd.isnull(Date) or pd.isnull(Symbol) or Market_Cap == 0:
                continue
            self.currency_data[Symbol].append(np.array([self.currencies[Symbol] * Close, self.asset_value, Open, High,
                                                        Low, Close, Volume, Market_Cap, self.dollars]))
            self.assets_daily[Date][Symbol] = Close

        for Symbol, faulty in faulty_currency.items():
            if self.currencies.get(Symbol) is not None:
                self.asset_value -= self.currencies[Symbol] * self.currency_data[Symbol][0][state.Close.value]
                del self.currency_data[Symbol]
                #del self.currencies[Symbol]

        self.currency_values = list(self.currencies.values())
        self.currency_keys = list(self.currencies.keys())
        self.dates = list(self.assets_daily.keys())
        self.initial_asset_value = self.asset_value
        self.action_space = 1
        self.observation_space = len(state)
        self.reset()

    def reset(self):
        self.iter = 0
        currency = random.randint(0, len(self.currency_data) - 1)
        self.sel_currency = self.currency_keys[currency]
        self.curr_state = self.currency_data[self.sel_currency][0]
        self.curr_state[state.Coin_Value.value] = self.currencies[self.sel_currency] *\
                                                  self.curr_state[state.Close.value]
        self.curr_state[state.Asset_Value.value] = self.asset_value
        self.dollars = random.randint(1000, 20000)
        self.curr_state[state.Dollars.value] = self.dollars

        return self.curr_state

    def step(self, action):
        trade_amount = action[0] * self.currencies[self.sel_currency]
        trade_val_bounded = min(self.dollars, trade_amount * self.curr_state[state.Close.value])
        trade_amount = trade_val_bounded / self.curr_state[state.Close.value]
        old_currency = self.currencies[self.sel_currency]
        self.currencies[self.sel_currency] += trade_amount
        self.dollars -= trade_val_bounded
        self.iter += 1
        next_state, done, info = self.currency_data[self.sel_currency][self.iter], False, {}
        next_state[state.Coin_Value.value] = self.currencies[self.sel_currency] * next_state[state.Close.value]
        next_state[state.Dollars.value] = self.dollars
        reward = (self.currencies[self.sel_currency] * next_state[state.Close.value] + trade_val_bounded) -\
                 old_currency * next_state[state.Close.value]
        new_asset_value = 0

        for Symbol, Close in self.assets_daily[self.dates[self.iter]].items():
            new_asset_value += self.currencies[Symbol] * Close

        next_state[state.Asset_Value.value] = new_asset_value
        self.curr_state = next_state
        self.asset_value = new_asset_value

        if self.iter == len(self.currency_data[self.sel_currency]) - 1:
            done = True

        return next_state, reward, done, info

    def render(self):
        value_made = self.asset_value - self.initial_asset_value
        print('Money made: ' + str(float(value_made)) + '$')

class state(Enum):
    Coin_Value = 0
    Asset_Value = 1
    Open = 2
    High = 3
    Low = 4
    Close = 5
    Volume = 6
    Market_Cap = 7
    Dollars = 8