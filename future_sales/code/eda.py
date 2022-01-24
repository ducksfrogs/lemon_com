import numpy as np
import pandas as pd
import random as rd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


#settings
import warnings
warnings.filterwarnings("ignore")


sales = pd.read_csv("../../data/sales_train.csv")

item_cat = pd.read_csv("../../data/item_categories.csv")
item = pd.read_csv("../../data/items.csv")
sub = pd.read_csv("../../data/sample_submission.csv")
shops = pd.read_csv("../../data/shops.csv")
test = pd.read_csv("../../data/test.csv")


sales.date = sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))


monthly_sales = sales.groupby(["date_block_num", "shop_id", "item_id"])[
                "date", "item_price", "item_cnt_day"].agg({"date":["min", "max"],
                                        "item_price": "mean", "item_cnt_day": "sum"})
