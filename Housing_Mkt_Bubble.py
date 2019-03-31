# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 02:36:10 2019

@author: achow
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:02:57 2019

@author: achow
"""

import pandas as pd

# set formatting
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)




###################

#GOOD Rolling Window year may be outlier to detect housing crash
# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot


# Apply your custom function and plot
#prices_perc = prices.rolling(7).apply(percent_change)
#prices_perc.loc["2014":"2018"].plot(lw=1)
#plt.show()

##################

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import calendar
from time import strptime

ASPUS = pd.read_csv('C:/scripts/capstone2/ASPUS.csv', index_col=0)

ASPUS.index = pd.to_datetime(ASPUS.index)
ASPUS = ASPUS.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
ASPUS.head(20)


ASPUS = ASPUS.interpolate(method='linear')
ASPUS.head(20)

ASPUS.tail(36)

ASPUS['ASPUS_PCT_CHG']= ASPUS['ASPUS_M'].pct_change()
ASPUS.head(30)
type(ASPUS['ASPUS_PCT_CHG'])
#prices_perc = aapl.rolling(7).apply(percent_change)
ASPUS['ASPUS_Q_PCT_CHG']= ASPUS['ASPUS_M'].pct_change(periods = 2)
ASPUS['ASPUS_A_PCT_CHG']= ASPUS['ASPUS_M'].pct_change(periods = 11)
ASPUS['ASPUS_2A_PCT_CHG']= ASPUS['ASPUS_M'].pct_change(periods = 23)
ASPUS['ASPUS_3A_PCT_CHG']= ASPUS['ASPUS_M'].pct_change(periods = 35)

ASPUS = ASPUS.fillna(method='bfill')

# Print out the number of missing values
print(ASPUS.isnull().sum())
ASPUS.info()

ASPUS['1-1-2004':'1-1-2007']
ASPUS['1-1-2007':'1-1-2010']

ASPUS_DROP= ASPUS[ASPUS['ASPUS_3A_PCT_CHG'] < .35]
print(ASPUS_DROP.tail(5))
ASPUS_DROP[['ASPUS_2A_PCT_CHG', 'ASPUS_3A_PCT_CHG']].plot()

ASPUS_DROP[['ASPUS_A_PCT_CHG','ASPUS_2A_PCT_CHG', 'ASPUS_3A_PCT_CHG']].plot()

ASPUS.to_csv('C:/scripts/capstone2/ASPUS2.csv')
#ASPUS.to_csv('C:/scripts/capstone2/ASPUS.csv')

ASPUS['ASPUS_A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 12 Months ",legend=True)


ASPUS['ASPUS_2A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE in 24 Months ",legend=True)

#Good
ASPUS['ASPUS_3A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 36 Months ",legend=True)

#Very Good
ASPUS['ASPUS_M'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 24 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US AVE HOUSE PURCHASE PRICE in 12, 24 & 36 Months')
plt.show()
###################

RPCE_M = pd.read_csv('C:/scripts/capstone2/RPCE_M.csv', index_col=0)

RPCE_M.index = pd.to_datetime(RPCE_M.index)
RPCE_M = RPCE_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
RPCE_M.head(20)

RPCE_M = RPCE_M.interpolate(method='linear')
RPCE_M.head(20)

RPCE_M.tail(36)

RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M2.csv')

RPCE_M.tail()
#RPCE_M = pd.read_csv('C:/scripts/capstone2/RPCE_M2.csv', index_col=0)
#ASPUS_M.head(20)
RPCE_M['RPCE_M_PCT_CHG']= RPCE_M['RPCE_M'].pct_change()
RPCE_M.head(30)
RPCE_M.tail(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
RPCE_M['RPCE_Q_PCT_CHG']= RPCE_M['RPCE_M'].pct_change(periods = 2)
RPCE_M['RPCE_A_PCT_CHG']= RPCE_M['RPCE_M'].pct_change(periods = 11)
RPCE_M['RPCE_2A_PCT_CHG']= RPCE_M['RPCE_M'].pct_change(periods = 23)
RPCE_M['RPCE_3A_PCT_CHG']= RPCE_M['RPCE_M'].pct_change(periods = 35)

RPCE_M = RPCE_M.fillna(method='bfill')


RPCE_M.describe()

#REmoving Outliers
RPCE_M.loc[RPCE_M.RPCE_M_PCT_CHG < -2.2, 'RPCE_M_PCT_CHG'] = np.nan
RPCE_M.loc[RPCE_M.RPCE_M_PCT_CHG > 2.2, 'RPCE_M_PCT_CHG'] = np.nan

RPCE_M.loc[RPCE_M.RPCE_Q_PCT_CHG < -4.3, 'RPCE_Q_PCT_CHG'] = np.nan
RPCE_M.loc[RPCE_M.RPCE_Q_PCT_CHG > 4.7, 'RPCE_Q_PCT_CHG'] = np.nan

RPCE_M.loc[RPCE_M.RPCE_A_PCT_CHG < -5.9, 'RPCE_A_PCT_CHG'] = np.nan
RPCE_M.loc[RPCE_M.RPCE_A_PCT_CHG > 5.8, 'RPCE_A_PCT_CHG'] = np.nan

RPCE_M.loc[RPCE_M.RPCE_2A_PCT_CHG < -4.1, 'RPCE_2A_PCT_CHG'] = np.nan
RPCE_M.loc[RPCE_M.RPCE_2A_PCT_CHG > 7.9, 'RPCE_2A_PCT_CHG'] = np.nan

RPCE_M.loc[RPCE_M.RPCE_3A_PCT_CHG < -8.8, 'RPCE_3A_PCT_CHG'] = np.nan
RPCE_M.loc[RPCE_M.RPCE_3A_PCT_CHG > 7.9, 'RPCE_3A_PCT_CHG'] = np.nan


RPCE_M.head(40)
RPCE_M.tail(60)
RPCE_M = RPCE_M.interpolate(method='linear')
RPCE_M = RPCE_M.fillna(method='bfill')

print(RPCE_M.isnull().sum())
RPCE_M.info()

# Print out the number of missing values


RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M2.csv')
#RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M.csv')

RPCE_M['RPCE_Q_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 12 Months ",legend=True)



RPCE_M['RPCE_A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 12 Months ",legend=True)

RPCE_M['RPCE_2A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 24 Months ",legend=True)
#Good
RPCE_M['RPCE_3A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 36 Months ",legend=True)

RPCE_M['RPCE_M'].plot(title="US Real People Consumer Expenditure Price Percent Change in 36 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Real People Consumer Expenditure Price Percent Change in 12, 24 & 36 Months')
plt.show()
##################
BAA_YEILD_10Y_M = pd.read_csv('C:/scripts/capstone2/BAA_YEILD_10Y_M.csv', index_col=0)

BAA_YEILD_10Y_M.index = pd.to_datetime(BAA_YEILD_10Y_M.index)
BAA_YEILD_10Y_M = BAA_YEILD_10Y_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))

BAA_YEILD_10Y_M.head(20)

BAA_YEILD_10Y_M = BAA_YEILD_10Y_M.interpolate(method='linear')
BAA_YEILD_10Y_M.head(20)

BAA_YEILD_10Y_M.tail(36)

#RPCE_M=RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M2.csv')


BAA_YEILD_10Y_M['BAA_YEILD_10Y_M_PCT_CHG']= BAA_YEILD_10Y_M['BAA10YM'].pct_change()
BAA_YEILD_10Y_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
BAA_YEILD_10Y_M['BAA_YEILD_10Y_Q_PCT_CHG']= BAA_YEILD_10Y_M['BAA10YM'].pct_change(periods = 2)
BAA_YEILD_10Y_M['BAA_YEILD_10Y_A_PCT_CHG']= BAA_YEILD_10Y_M['BAA10YM'].pct_change(periods = 11)
BAA_YEILD_10Y_M['BAA_YEILD_10Y_2A_PCT_CHG']= BAA_YEILD_10Y_M['BAA10YM'].pct_change(periods = 23)
BAA_YEILD_10Y_M['BAA_YEILD_10Y_3A_PCT_CHG']= BAA_YEILD_10Y_M['BAA10YM'].pct_change(periods = 35)


BAA_YEILD_10Y_M = BAA_YEILD_10Y_M.fillna(method='bfill')

# Print out the number of missing values
print(BAA_YEILD_10Y_M.isnull().sum())
BAA_YEILD_10Y_M.info()

BAA_YEILD_10Y_M.to_csv('C:/scripts/capstone2/BAA_YEILD_10Y_M2.csv')
BAA_YEILD_10Y_M.to_csv('C:/scripts/capstone2/BAA_YEILD_10Y_M.csv')

BAA_YEILD_10Y_M['BAA_YEILD_10Y_Q_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 3 Months ",legend=True)

BAA_YEILD_10Y_M['BAA_YEILD_10Y_A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 12 Months ",legend=True)

#Good
BAA_YEILD_10Y_M['BAA_YEILD_10Y_2A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 24 Months ",legend=True)

BAA_YEILD_10Y_M['BAA_YEILD_10Y_3A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 36 Months ",legend=True)

#Good
BAA_YEILD_10Y_M['BAA10YM'].plot(title="BAA Bond 10Y yealds Percent Change in 1 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('BAA Bond 10Y yealds Percent Change in 24 Months in 12, 24 & 36 Months')
plt.show()
####################

UEMP_Q = pd.read_csv('C:/scripts/capstone2/UEMP_Q.csv', index_col=0)

UEMP_Q.index = pd.to_datetime(UEMP_Q.index)
UEMP_M = UEMP_Q.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
UEMP_M.head(20)


UEMP_M = UEMP_M.interpolate(method='linear')
UEMP_M.head(20)

UEMP_M.tail(36)

#RPCE_M=RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M2.csv')

#Data Cleaning & Removing outliers for
UEMP_M['UEMP_M_PCT_CHG']= UEMP_M['LRUN_UEMP'].pct_change()
UEMP_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
UEMP_M['UEMP_Q_PCT_CHG']= UEMP_M['LRUN_UEMP'].pct_change(periods = 2)
UEMP_M['UEMP_A_PCT_CHG']= UEMP_M['LRUN_UEMP'].pct_change(periods = 11)
UEMP_M['UEMP_2A_PCT_CHG']= UEMP_M['LRUN_UEMP'].pct_change(periods = 23)
UEMP_M['UEMP_3A_PCT_CHG']= UEMP_M['LRUN_UEMP'].pct_change(periods = 35)

UEMP_M = UEMP_M.fillna(method='bfill')

# Print out the number of missing values
print(UEMP_M.isnull().sum())
UEMP_M.info()

UEMP_M['1-1-2004':'1-1-2007']
UEMP_M['1-1-2007':'1-1-2010']

UEMP_JUMP= UEMP_M[UEMP_M['UEMP_3A_PCT_CHG'] > .35]
print(UEMP_JUMP.tail(50))
UEMP_JUMP[['UEMP_2A_PCT_CHG', 'UEMP_3A_PCT_CHG']].plot()

UEMP_JUMP[['UEMP_A_PCT_CHG','UEMP_2A_PCT_CHG', 'UEMP_3A_PCT_CHG']].plot()

UEMP_M.to_csv('C:/scripts/capstone2/UEMP_M2.csv')
UEMP_M.to_csv('C:/scripts/capstone2/UEMP_M.csv')

UEMP_M['UEMP_Q_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 3 Months ",legend=True)

UEMP_M['UEMP_A_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 12 Months ",legend=True)


UEMP_M['UEMP_2A_PCT_CHG'].plot(title="US Long Run Unemployment in 24 Months ",legend=True)

#Good
UEMP_M['UEMP_3A_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 36 Months ",legend=True)

#Very Good
UEMP_M['LRUN_UEMP'].plot(title="US Long Run Unemployment Percent Change in 24 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Long Run Unemployment in 12, 24 & 36 Months')
plt.show()


####################

RGDP_Q = pd.read_csv('C:/scripts/capstone2/RGDP_Q.csv', index_col=0)

RGDP_Q.index = pd.to_datetime(RGDP_Q.index)
RGDP_M = RGDP_Q.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
RGDP_M.head(20)


RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
RGDP_M.head(20)

RGDP_M.tail(36)

RGDP_M['GDP_M']=RGDP_M['GDP_M']
RGDP_M['RGDP_M_PCT_CHG']= RGDP_M['GDP_M'].pct_change()
RGDP_M.head(30)
RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
#prices_perc = aapl.rolling(7).apply(percent_change)
RGDP_M['RGDP_Q_PCT_CHG']= RGDP_M['GDP_M'].pct_change(periods = 2)
RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
#Good
RGDP_M['RGDP_A_PCT_CHG']= RGDP_M['GDP_M'].pct_change(periods = 11)
RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
RGDP_M['RGDP_2A_PCT_CHG']= RGDP_M['GDP_M'].pct_change(periods = 23)
RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
RGDP_M['RGDP_3A_PCT_CHG']= RGDP_M['GDP_M'].pct_change(periods = 35)

RGDP_M = RGDP_M.fillna(method='bfill')

# Print out the number of missing values
print(RGDP_M.isnull().sum())
RGDP_M.info()

#Outlier
RGDP_M.describe()
from numpy import inf
RGDP_M=RGDP_M.replace(-inf, np.nan)
RGDP_M=RGDP_M.replace(inf, np.nan)
RGDP_M = RGDP_M.interpolate(method='linear')
RGDP_M = RGDP_M.fillna(method='bfill')

#x[x == -inf] = 0

#RGDP_M[RGDP_M.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

RGDP_M.loc[RGDP_M.RGDP_M_PCT_CHG == 0, 'RGDP_M_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_Q_PCT_CHG == 0, 'RGDP_Q_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_A_PCT_CHG == 0, 'RGDP_A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_2A_PCT_CHG == 0, 'RGDP_2A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_3A_PCT_CHG == 0, 'RGDP_3A_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_M_PCT_CHG == inf, 'RGDP_M_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_Q_PCT_CHG == inf, 'RGDP_Q_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_A_PCT_CHG == inf, 'RGDP_A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_2A_PCT_CHG == inf, 'RGDP_2A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_3A_PCT_CHG == inf, 'RGDP_3A_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_M_PCT_CHG == -inf, 'RGDP_M_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_Q_PCT_CHG == -inf, 'RGDP_Q_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_A_PCT_CHG == -inf, 'RGDP_A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_2A_PCT_CHG == -inf, 'RGDP_2A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_3A_PCT_CHG == -inf, 'RGDP_3A_PCT_CHG'] = np.nan

print(RGDP_M.isnull().sum())
RGDP_M.info()


RGDP_M = RGDP_M.interpolate(method='linear')

RGDP_M = RGDP_M.fillna(method='bfill')
RGDP_M.describe()

RGDP_M.describe()
RGDP_M.loc[RGDP_M.RGDP_M_PCT_CHG < -3.2, 'RGDP_M_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_M_PCT_CHG > 1.75, 'RGDP_M_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_Q_PCT_CHG < -3.2, 'RGDP_Q_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_Q_PCT_CHG > 3.6, 'RGDP_Q_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_A_PCT_CHG < -5.7, 'RGDP_A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_A_PCT_CHG > 5.1, 'RGDP_A_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_2A_PCT_CHG < -5.3, 'RGDP_2A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_2A_PCT_CHG > 6.3, 'RGDP_2A_PCT_CHG'] = np.nan

RGDP_M.loc[RGDP_M.RGDP_3A_PCT_CHG < -5.20, 'RGDP_3A_PCT_CHG'] = np.nan
RGDP_M.loc[RGDP_M.RGDP_3A_PCT_CHG > 5.0, 'RGDP_3A_PCT_CHG'] = np.nan


RGDP_M.head(40)
RGDP_M.tail(60)
RGDP_M = RGDP_M.interpolate(method='linear')
RGDP_M = RGDP_M.fillna(method='bfill')

print(RGDP_M.isnull().sum())
RGDP_M.info()
RGDP_M.describe()

RGDP_M.to_csv('C:/scripts/capstone2/RGDP_M2.csv')
#RGDP_M.to_csv('C:/scripts/capstone2/RGDP_M.csv')

#Good
RGDP_M['RGDP_M_PCT_CHG'].plot(title="US GDP Percent Change in 3 Months ",legend=True)

RGDP_M['RGDP_Q_PCT_CHG'].plot(title="US GDP Percent Change in 3 Months ",legend=True)

RGDP_M['RGDP_A_PCT_CHG'].plot(title="US GDP Percent Change in 12 Months ",legend=True)

RGDP_M['RGDP_2A_PCT_CHG'].plot(title="US GDP in 24 Months ",legend=True)

RGDP_M['RGDP_3A_PCT_CHG'].plot(title="US GDP Percent Change in 36 Months ",legend=True)

#Very Good
RGDP_M['GDP_M'].plot(title="US GDP Percent Change in 24 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US GDP in 12, 24 & 36 Months')
plt.show()


####################


US10Y_M = pd.read_csv('C:/scripts/capstone2/US10Y_M.csv', index_col=0)

US10Y_M.index = pd.to_datetime(US10Y_M.index)
US10Y_M = US10Y_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
US10Y_M.head(20)


US10Y_M = US10Y_M.interpolate(method='linear')
US10Y_M.head(20)

US10Y_M['1-1-2004':'1-1-2007']
US10Y_M['1-1-2007':'1-1-2010']

print([US10Y_M['US10Y_3A_PCT_CHG'] > .18])

RATE_JUMP= US10Y_M[US10Y_M['US10Y_3A_PCT_CHG'] > .3]
print(RATE_JUMP.tail(50))
RATE_JUMP[['US10Y_A_PCT_CHG','US10Y_2A_PCT_CHG', 'US10Y_3A_PCT_CHG']].plot()

US10Y_M['US10Y_M_PCT_CHG']= US10Y_M['US10Y_M'].pct_change()
US10Y_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
US10Y_M['US10Y_Q_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 2)
US10Y_M['US10Y_A_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 11)
US10Y_M['US10Y_2A_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 23)
US10Y_M['US10Y_3A_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 35)

US10Y_M = US10Y_M.fillna(method='bfill')

# Print out the number of missing values
print(US10Y_M.isnull().sum())
US10Y_M.info()

US10Y_M.to_csv('C:/scripts/capstone2/US10Y_M2.csv')
#US10Y_M.to_csv('C:/scripts/capstone2/US10Y_M.csv')

US10Y_M['US10Y_M'].plot(title="US 10 Year Treasury Monthly Rates ",legend=True)

US10Y_M['US10Y_M_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 12 Months ",legend=True)

US10Y_M['US10Y_Q_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 3 Months ",legend=True)

US10Y_M['US10Y_A_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 12 Months ",legend=True)


US10Y_M['US10Y_2A_PCT_CHG'].plot(title="US 10 Year Treasury in 24 Months ",legend=True)

#Good
US10Y_M['US10Y_3A_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 35)
US10Y_M['US10Y_3A_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 36 Months ",legend=True)


plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US 10 Year Treasury in 12, 24 & 36 Months')
plt.show()

####################

STOCK_MKT_Q = pd.read_csv('C:/scripts/capstone2/STOCK_MKT_Q.csv', index_col=0)

STOCK_MKT_Q.index = pd.to_datetime(STOCK_MKT_Q.index)
STOCK_MKT_M = STOCK_MKT_Q.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
STOCK_MKT_M.head(20)


STOCK_MKT_M = STOCK_MKT_M.interpolate(method='linear')
STOCK_MKT_M.head(20)

STOCK_MKT_M.tail(36)

STOCK_MKT_M['STOCK_MKT_M_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 1)
STOCK_MKT_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
STOCK_MKT_M['STOCK_MKT_Q_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 2)
STOCK_MKT_M['STOCK_MKT_A_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 11)
STOCK_MKT_M['STOCK_MKT_2A_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 23)
STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 35)

STOCK_MKT_M = STOCK_MKT_M.fillna(method='bfill')

# Print out the number of missing values
print(STOCK_MKT_M.isnull().sum())
STOCK_MKT_M.info()

#Replace Outliers

#median = STOCK_MKT_M.loc[STOCK_MKT_M['STOCK_MKT_M_PCT_CHG'] < -10, 'STOCK_MKT_M_PCT_CHG'].median()
STOCK_MKT_M.describe()

STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_M_PCT_CHG < -1.93, 'STOCK_MKT_M_PCT_CHG'] = np.nan
STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_M_PCT_CHG > 1.93, 'STOCK_MKT_M_PCT_CHG'] = np.nan

STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_Q_PCT_CHG < -4.82, 'STOCK_MKT_Q_PCT_CHG'] = np.nan
STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_Q_PCT_CHG > 4.82, 'STOCK_MKT_Q_PCT_CHG'] = np.nan

STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_A_PCT_CHG < -6.39, 'STOCK_MKT_A_PCT_CHG'] = np.nan
STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_A_PCT_CHG > 6.0, 'STOCK_MKT_A_PCT_CHG'] = np.nan

STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_2A_PCT_CHG < -8.5, 'STOCK_MKT_2A_PCT_CHG'] = np.nan
STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_2A_PCT_CHG > 8.5, 'STOCK_MKT_2A_PCT_CHG'] = np.nan

STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_3A_PCT_CHG < -10.0, 'STOCK_MKT_3A_PCT_CHG'] = np.nan
STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_3A_PCT_CHG > 10.0, 'STOCK_MKT_3A_PCT_CHG'] = np.nan


STOCK_MKT_M.head(40)
STOCK_MKT_M.tail(60)
STOCK_MKT_M = STOCK_MKT_M.interpolate(method='linear')
STOCK_MKT_M = STOCK_MKT_M.fillna(method='bfill')

STOCK_MKT_M.describe()

print(STOCK_MKT_M.isnull().sum())
STOCK_MKT_M.info()

#
#median = STOCK_MKT_M.loc[STOCK_MKT_M['STOCK_MKT_2A_PCT_CHG'] < 1, 'STOCK_MKT_2A_PCT_CHG'].median()
#STOCK_MKT_M.loc[STOCK_MKT_M.STOCK_MKT_2A_PCT_CHG > 1, 'STOCK_MKT_2A_PCT_CHG'] = np.nan

STOCK_MKT_M['1-1-2004':'1-1-2007']
STOCK_MKT_M['1-1-2007':'1-1-2010']

STOCK_MKT_DROP= STOCK_MKT_M[STOCK_MKT_M['STOCK_MKT_A_PCT_CHG'] < -5]
print(STOCK_MKT_DROP.tail(50))
STOCK_MKT_DROP[[ 'STOCK_MKT_2A_PCT_CHG', 'STOCK_MKT_3A_PCT_CHG']].plot()

STOCK_MKT_DROP[['STOCK_MKT_A_PCT_CHG','STOCK_MKT_2A_PCT_CHG', 'STOCK_MKT_3A_PCT_CHG']].plot()

STOCK_MKT_M.to_csv('C:/scripts/capstone2/STOCK_MKT_M2.csv')
STOCK_MKT_M.to_csv('C:/scripts/capstone2/STOCK_MKT_M.csv')

STOCK_MKT_M['STOCK_MKT_Q_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 3 Months ",legend=True)

#Very Good
STOCK_MKT_M['STOCK_MKT_A_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 12 Months ",legend=True)


STOCK_MKT_M['STOCK_MKT_2A_PCT_CHG'].plot(title="US STOCK MARKET in 24 Months ",legend=True)

#Good
STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG']=STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG']
STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 36 Months ",legend=True)

#Very Good
STOCK_MKT_M['STOCK_MKT_M'].plot(title="US STOCK MARKET Percent Change in 1 Month ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US STOCK MARKET in 12, 24 & 36 Months')
plt.show()

####################



####################
H_SUPPLY_M = pd.read_csv('C:/scripts/capstone2/H_SUPPLY_M.csv', index_col=0)

H_SUPPLY_M.index = pd.to_datetime(H_SUPPLY_M.index)
H_SUPPLY_M = H_SUPPLY_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
H_SUPPLY_M.head(20)


H_SUPPLY_M = H_SUPPLY_M.interpolate(method='linear')
H_SUPPLY_M.head(20)

H_SUPPLY_M.tail(36)

H_SUPPLY_M['H_RATIO_M_PCT_CHG']= H_SUPPLY_M['H_RATIO_M'].pct_change()
H_SUPPLY_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
H_SUPPLY_M['H_RATIO_Q_PCT_CHG']= H_SUPPLY_M['H_RATIO_M'].pct_change(periods = 2)
H_SUPPLY_M['H_RATIO_A_PCT_CHG']= H_SUPPLY_M['H_RATIO_M'].pct_change(periods = 11)
H_SUPPLY_M['H_RATIO_2A_PCT_CHG']= H_SUPPLY_M['H_RATIO_M'].pct_change(periods = 23)
H_SUPPLY_M['H_RATIO_3A_PCT_CHG']= H_SUPPLY_M['H_RATIO_M'].pct_change(periods = 35)

H_SUPPLY_M = H_SUPPLY_M.interpolate(method='linear')

H_SUPPLY_M = H_SUPPLY_M.fillna(method='bfill')

# Print out the number of missing values
print(H_SUPPLY_M.isnull().sum())
H_SUPPLY_M.info()

H_SUPPLY_M['1-1-2004':'1-1-2007']
H_SUPPLY_M['1-1-2007':'1-1-2010']

H_RATIO_JUMP= H_SUPPLY_M[H_SUPPLY_M['H_RATIO_3A_PCT_CHG'] > .35]
print(H_RATIO_JUMP.tail(50))
H_RATIO_JUMP[['H_RATIO_2A_PCT_CHG', 'H_RATIO_3A_PCT_CHG']].plot()

H_RATIO_JUMP[['H_RATIO_A_PCT_CHG','H_RATIO_2A_PCT_CHG', 'H_RATIO_3A_PCT_CHG']].plot()

H_SUPPLY_M.to_csv('C:/scripts/capstone2/H_SUPPLY_M2.csv')
H_SUPPLY_M.to_csv('C:/scripts/capstone2/H_SUPPLY_M.csv')

H_SUPPLY_M['H_RATIO_A_PCT_CHG'].plot(title="US Housing Supply RATIO Percent Change in 12 Months ",legend=True)


H_SUPPLY_M['H_RATIO_2A_PCT_CHG'].plot(title="US Housing Supply RATIO in 24 Months ",legend=True)

#Good
H_SUPPLY_M['H_RATIO_3A_PCT_CHG'].plot(title="US Housing Supply RATIO Percent Change in 36 Months ",legend=True)

#Very Good
H_SUPPLY_M['H_RATIO_M'].plot(title="US Housing Supply RATIO Percent Change in 24 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Supply RATIO in 12, 24 & 36 Months')
plt.show()


######################
HSN1F_M = pd.read_csv('C:/scripts/capstone2/HSN1F_M.csv', index_col=0)

HSN1F_M.index = pd.to_datetime(HSN1F_M.index)
HSN1F_M = HSN1F_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
HSN1F_M.head(20)


HSN1F_M = HSN1F_M.interpolate(method='linear')
HSN1F_M.head(20)

HSN1F_M.tail(36)

HSN1F_M['HSN1F_M_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change()
HSN1F_M.head(3)

#prices_perc = aapl.rolling(7).apply(percent_change)
HSN1F_M['HSN1F_Q_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change(periods = 2)
HSN1F_M['HSN1F_A_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change(periods = 11)
HSN1F_M['HSN1F_2A_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change(periods = 23)
HSN1F_M['HSN1F_3A_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change(periods = 35)

HSN1F_M = HSN1F_M.fillna(method='bfill')

# Print out the number of missing values
print(HSN1F_M.isnull().sum())
HSN1F_M.info()

HSN1F_M['1-1-2004':'1-1-2007']
HSN1F_M['1-1-2007':'1-1-2010']

HSN1F_M['1-1-2014':'12-1-2018']

HSN1F_JUMP= HSN1F_M[HSN1F_M['HSN1F_3A_PCT_CHG'] > .35]
print(HSN1F_JUMP.tail(50))
HSN1F_JUMP[['HSN1F_2A_PCT_CHG', 'HSN1F_3A_PCT_CHG']].plot()

HSN1F_JUMP[['HSN1F_A_PCT_CHG','HSN1F_2A_PCT_CHG', 'HSN1F_3A_PCT_CHG']].plot()

HSN1F_M.to_csv('C:/scripts/capstone2/HSN1F_M2.csv')
#HSN1F_M.to_csv('C:/scripts/capstone2/HSN1F_M.csv')

HSN1F_M['HSN1F_A_PCT_CHG'].plot(title="US New 1F Housing Supply Percent Change in 12 Months ",legend=True)


HSN1F_M['HSN1F_2A_PCT_CHG'].plot(title="US New 1F Housing Supply in 24 Months ",legend=True)

#Good
HSN1F_M['HSN1F_3A_PCT_CHG'].plot(title="US New 1F Housing Supply Percent Change in 36 Months ",legend=True)

#Very Good
HSN1F_M['HSN1F_M'].plot(title="US New 1F Housing Supply ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US New 1F Housing Supplyin 12, 24 & 36 Months')
plt.show()

########################

PERMIT_M = pd.read_csv('C:/scripts/capstone2/PERMIT_M.csv', index_col=0)

PERMIT_M.index = pd.to_datetime(PERMIT_M.index)
PERMIT_M = PERMIT_M.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
PERMIT_M.head(20)


PERMIT_M = PERMIT_M.interpolate(method='linear')
PERMIT_M.head(20)

PERMIT_M.tail(36)

PERMIT_M['PERMIT_M_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change()
PERMIT_M.head(30)

#prices_perc = aapl.rolling(7).apply(percent_change)
PERMIT_M['PERMIT_Q_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change(periods = 2)
PERMIT_M['PERMIT_A_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change(periods = 11)
PERMIT_M['PERMIT_2A_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change(periods = 23)
PERMIT_M['PERMIT_3A_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change(periods = 35)


PERMIT_M = PERMIT_M.interpolate(method='linear')

PERMIT_M = PERMIT_M.fillna(method='bfill')

# Print out the number of missing values
print(PERMIT_M.isnull().sum())
PERMIT_M.info()

PERMIT_M['1-1-2004':'1-1-2007']
PERMIT_M['1-1-2007':'1-1-2010']
PERMIT_M['1-1-2016':'12-1-2018']

PERMIT_DROP= PERMIT_M[PERMIT_M['PERMIT_3A_PCT_CHG'] < .35]
print(PERMIT_DROP.tail(50))
PERMIT_DROP[['PERMIT_2A_PCT_CHG', 'PERMIT_3A_PCT_CHG']].plot()

PERMIT_DROP[['PERMIT_A_PCT_CHG','PERMIT_2A_PCT_CHG', 'PERMIT_3A_PCT_CHG']].plot()

PERMIT_M.to_csv('C:/scripts/capstone2/PERMIT_M2.csv')
#PERMIT_M.to_csv('C:/scripts/capstone2/PERMIT_M.csv')

PERMIT_M['PERMIT_A_PCT_CHG'].plot(title="US Construction PERMIT Percent Change in 12 Months ",legend=True)


PERMIT_M['PERMIT_2A_PCT_CHG'].plot(title="US Construction PERMIT in 24 Months ",legend=True)

#Good
PERMIT_M['PERMIT_3A_PCT_CHG'].plot(title="US Construction PERMIT Percent Change in 36 Months ",legend=True)

#Very Good
PERMIT_M['PERMIT_M'].plot(title="US Construction PERMIT Percent Change in 24 Months ",legend=True)
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Construction PERMIT in 12, 24 & 36 Months')
plt.show()
########################

#####################

ASPUS['ASPUS_3A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 36 Months ",fontsize=12,  linewidth=3.8, legend=True)

#STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 12 Months ",legend=True)

STOCK_MKT_M['STOCK_MKT_M'].plot(title="US STOCK MARKET Percent Change in 1 Month ",fontsize=12,  linewidth=1.8, legend=True)

BAA_YEILD_10Y_M['BAA_YEILD_10Y_2A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 24 Months ",fontsize=12,  linewidth=2.8, legend=True)

US10Y_M['US10Y_3A_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 36 Months ",fontsize=12,  linewidth=3, legend=True)

#plt.yscale('log')
#plt.yscale('logit')
#plt.yscale('symlog', linthreshy=0.5)
#plt.yscale('linear')
plt.yscale('symlog',basey=3)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Stock, Bond and 10YT')

plt.show()

#########################

ASPUS['ASPUS_3A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 36 Months ", fontsize=12,  linewidth=4.2, legend=True)

RPCE_M['RPCE_A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 12 Months ", fontsize=12,  linewidth=2.8, legend=True)

UEMP_M['UEMP_3A_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 36 Months ", fontsize=12,  linewidth=3,legend=True)

RGDP_M['GDP_M'].plot(title="US GDP Percent Change in 24 Months " , color = 'black', fontsize=22,  linewidth=1.8,legend=True)

plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes', fontsize=12)
plt.title('US Housing Price vs Consumer Expenditure, UEMP and GDP', fontsize=14)

plt.show()


#########################

ASPUS['ASPUS_3A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 36 Months ",fontsize=12,  linewidth=4.2, legend=True)

H_SUPPLY_M['H_RATIO_3A_PCT_CHG'].plot(title="US Housing Supply RATIO Percent Change in 36 Months ",fontsize=12,  linewidth=2.2, legend=True)

HSN1F_M['HSN1F_3A_PCT_CHG'].plot(title="US New 1F Housing Supply Percent Change in 36 Months ",fontsize=12,  linewidth=2.0, legend=True)

PERMIT_M['PERMIT_3A_PCT_CHG'].plot(title="US Construction PERMIT Percent Change in 36 Months ",fontsize=12,  linewidth=1.8, legend=True)

plt.xlabel('Date')

plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Housing Supply Ratio, New 1F Supply and Construction Permit')

plt.show()
################################

ASPUS['ASPUS_3A_PCT_CHG'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change in 36 Months ",fontsize=12,  linewidth=4.8, legend=True)

H_SUPPLY_M['H_RATIO_3A_PCT_CHG'].plot(title="US Housing Supply RATIO Percent Change in 36 Months ",fontsize=12,  linewidth=2.8, legend=True)

HSN1F_M['HSN1F_3A_PCT_CHG'].plot(title="US New 1F Housing Supply Percent Change in 36 Months ",fontsize=12,  linewidth=2.6, legend=True)

PERMIT_M['PERMIT_3A_PCT_CHG'].plot(title="US Construction PERMIT Percent Change in 36 Months ",fontsize=12,  linewidth=2.6, legend=True)


STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 12 Months ",fontsize=12,  linewidth=1.2, legend=True)

#STOCK_MKT_M['STOCK_MKT_M'].plot(title="US STOCK MARKET Percent Change in 1 Month ",legend=True)

BAA_YEILD_10Y_M['BAA_YEILD_10Y_2A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 24 Months ",fontsize=12,  linewidth=2.8, legend=True)

US10Y_M['US10Y_3A_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 36 Months ",fontsize=12,  linewidth=3, legend=True)


RPCE_M['RPCE_A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 12 Months ",fontsize=12,  linewidth=1.2, legend=True)

UEMP_M['UEMP_3A_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 36 Months ",fontsize=12,  linewidth=1.2, legend=True)

#RGDP_M['GDP_M'].plot(title="US GDP Percent Change in 24 Months ",legend=True)

RGDP_M['RGDP_M_PCT_CHG'].plot(title="US GDP Percent Change in 3 Months ",fontsize=12,  linewidth=1.2, legend=True)


plt.xlabel('Date')


plt.yscale('symlog',basey=2)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Stock, Bond and 10YT, GDP, UEMP, Housing data', fontsize=14)

plt.show()

###################### Reindex  #############
ASPUS['ASPUS_3A_PCT_CHG']= ASPUS['ASPUS_M'].pct_change(periods = 35)

HSN1F_M['HSN1F_3A_PCT_CHG']= HSN1F_M['HSN1F_M'].pct_change(periods = 35)
US10Y_M['US10Y_3A_PCT_CHG']= US10Y_M['US10Y_M'].pct_change(periods = 35)
RPCE_M['RPCE_A_PCT_CHG']= RPCE_M['RPCE_M'].pct_change(periods = 11)
PERMIT_M['PERMIT_3A_PCT_CHG']= PERMIT_M['PERMIT_M'].pct_change(periods = 35)
STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG']= STOCK_MKT_M['STOCK_MKT_M'].pct_change(periods = 35)


#####################################

ASPUS_3A=ASPUS[['ASPUS_3A_PCT_CHG']]

H_RATIO_3A=H_SUPPLY_M[['H_RATIO_3A_PCT_CHG']]
HSN1F_3A=HSN1F_M[['HSN1F_3A_PCT_CHG']]
PERMIT_3A=PERMIT_M[['PERMIT_3A_PCT_CHG']]


STOCK_MKT_3A=STOCK_MKT_M[['STOCK_MKT_3A_PCT_CHG']]
BAA_YEILD_10Y_2A=BAA_YEILD_10Y_M[['BAA_YEILD_10Y_2A_PCT_CHG']]
US10Y_3A=US10Y_M[['US10Y_3A_PCT_CHG']]


RPCE_A=RPCE_M[['RPCE_A_PCT_CHG']]
UEMP_3A=UEMP_M[['UEMP_3A_PCT_CHG']]
RGDP_M=RGDP_M[['RGDP_M_PCT_CHG']]

#result = left.join(right)
#result = left.join([right, right2])

housing_df = ASPUS_3A.join([H_RATIO_3A, HSN1F_3A, PERMIT_3A, STOCK_MKT_3A, 
                            BAA_YEILD_10Y_2A, US10Y_3A, 
                            RPCE_A, UEMP_3A, RGDP_M])

housing_df.info()

housing_df.describe()
plt.yscale('symlog',basey=1000000)
housing_df.plot(fontsize=12,  linewidth=4.2, legend=True)
plt.show()
print(housing_df.isnull().sum())

housing_df = housing_df.interpolate(method='linear')

housing_df = housing_df.fillna(method='bfill')
housing_df.describe()

housing_df.to_csv('C:/scripts/capstone2/housing_df.csv')
######################################
ASPUS_M.info()
ASPUS_M=ASPUS[['ASPUS_M']]
ASPUS_M.head()

H_RATIO_M=H_SUPPLY_M[['H_RATIO_M']]
HSN1F_M=HSN1F_M[['HSN1F_M']]
PERMIT_M=PERMIT_M[['PERMIT_M']]


STOCK_MKT_M=STOCK_MKT_M[['STOCK_MKT_M']]
BAA10YM=BAA_YEILD_10Y_M[['BAA10YM']]
US10Y_M=US10Y_M[['US10Y_M']]


RPCE_M=RPCE_M[['RPCE_M']]
LRUN_UEMP=UEMP_M[['LRUN_UEMP']]
GDP_M=RGDP_Q[['GDP_M']]

#result = left.join(right)
#result = left.join([right, right2])

h_m_df = ASPUS_M.join([H_RATIO_M, HSN1F_M, PERMIT_M, STOCK_MKT_M, BAA10YM, US10Y_M, RPCE_M, LRUN_UEMP, GDP_M])

h_m_df.info()

h_m_df.describe()
plt.yscale('symlog',basey=1000000)
h_m_df.plot(fontsize=12,  linewidth=4.2, legend=True)
plt.show()
print(h_m_df.isnull().sum())

h_m_df = h_m_df.interpolate(method='linear')

h_m_df = h_m_df.fillna(method='bfill')
h_m_df.describe()

h_m_df.to_csv('C:/scripts/capstone2/h_m_df.csv')


ASPUS_M_S = h_m_df[['ASPUS_M']]/10000
HSN1F_M_S = h_m_df[['HSN1F_M']]/100
PERMIT_M_S = h_m_df[['PERMIT_M']]/100

ASPUS_M_S.index = pd.to_datetime(ASPUS_M_S.index)
ASPUS_M_S = ASPUS_M_S.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))

HSN1F_M_S.index = pd.to_datetime(HSN1F_M_S.index)
HSN1F_M_S = HSN1F_M_S.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))

PERMIT_M_S.index = pd.to_datetime(PERMIT_M_S.index)
PERMIT_M_S = PERMIT_M_S.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))

h_m_s_df = ASPUS_M_S.join([H_RATIO_M, HSN1F_M_S, PERMIT_M_S, STOCK_MKT_M, BAA10YM, US10Y_M, RPCE_M, LRUN_UEMP, GDP_M])

h_m_s_df.plot(fontsize=12,  linewidth=3.2, legend=True, colormap='Dark2')

h_m_s_df = h_m_s_df.interpolate(method='linear')

h_m_s_df = h_m_s_df.fillna(method='bfill')
h_m_s_df.describe()

h_m_s_df.info()

########################    RAW DATA  PLOT HOUSING     ##########
ASPUS_M_S['ASPUS_M'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change  ",fontsize=12,  linewidth=4.8, legend=True)

H_SUPPLY_M['H_RATIO_M'].plot(title="US Housing Supply RATIO Percent Change ",fontsize=12,  linewidth=2.8, legend=True)

HSN1F_M_S['HSN1F_M'].plot(title="US New 1F Housing Supply Percent Change  ",fontsize=12,  linewidth=2.6, legend=True)

PERMIT_M_S['PERMIT_M'].plot(title="US Construction PERMIT Percent Change  ",fontsize=12,  linewidth=2.6, legend=True)

plt.xlabel('Date')

plt.yscale('symlog',basey=3)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Housing Supply Ration, New 1F Supply & Construction Permit', fontsize=14)

plt.show()
########################    RAW DATA  PLOT HOUSING vs MARKET DATA    ##########

ASPUS_M_S['ASPUS_M'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change  ",fontsize=12,  linewidth=4.8, legend=True)

BAA10YM['BAA10YM'].plot(title="BAA BONDS Percent Change  ",fontsize=12,  linewidth=3.2, legend=True)

US10Y_M['US10Y_M'].plot(title="US Treasury Rate Percent Change  ",fontsize=12,  linewidth=3.2, legend=True)

STOCK_MKT_M['STOCK_MKT_M'].plot(title="US Stock MarketPercent Change ",fontsize=12,  linewidth=1.8, legend=True)


plt.xlabel('Date')

#plt.yscale('symlog',basey=10)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs STOCK, BAA BONDS & Treasury Rate', fontsize=14)

plt.show()


########################    RAW DATA  PLOT HOUSING     ##########

ASPUS_M_S['ASPUS_M'].plot(title="US AVE HOUSE PURCHASE PRICE Percent Change  ",fontsize=12,  linewidth=4.8, legend=True)

RPCE_M['RPCE_M'].plot(title="US Real People Consumer Expenditure Price Percent Change",fontsize=12,  linewidth=1.8, legend=True)

LRUN_UEMP['LRUN_UEMP'].plot(title="Long Run Unemployment Percent Change  ",fontsize=12,  linewidth=3.2, legend=True)

GDP_M['GDP_M'].plot(title="US GDP Percent Change  ",fontsize=12,  linewidth=3.2, legend=True)

plt.xlabel('Date')

#plt.yscale('symlog',basey=10)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Consumer Expenditure , Unemployment & US GDP', fontsize=14)

plt.show()

##################################


##########################################################################

STOCK_MKT_M['STOCK_MKT_3A_PCT_CHG'].plot(title="US STOCK MARKET Percent Change in 12 Months ",fontsize=12,  linewidth=1.2, legend=True)

#STOCK_MKT_M['STOCK_MKT_M'].plot(title="US STOCK MARKET Percent Change in 1 Month ",legend=True)

BAA_YEILD_10Y_M['BAA_YEILD_10Y_2A_PCT_CHG'].plot(title="BAA Bond 10Y yealds Percent Change in 24 Months ",fontsize=12,  linewidth=2.8, legend=True)

US10Y_M['US10Y_3A_PCT_CHG'].plot(title="US 10 Year Treasury Percent Change in 36 Months ",fontsize=12,  linewidth=3, legend=True)


RPCE_M['RPCE_A_PCT_CHG'].plot(title="US Real People Consumer Expenditure Price Percent Change in 12 Months ",fontsize=12,  linewidth=1.2, legend=True)

UEMP_M['UEMP_3A_PCT_CHG'].plot(title="US Long Run Unemployment Percent Change in 36 Months ",fontsize=12,  linewidth=1.2, legend=True)

#RGDP_M['GDP_M'].plot(title="US GDP Percent Change in 24 Months ",legend=True)

RGDP_M['RGDP_M_PCT_CHG'].plot(title="US GDP Percent Change in 3 Months ",fontsize=12,  linewidth=1.2, legend=True)


plt.xlabel('Date')


plt.yscale('symlog',basey=2)
# where basex or basey are the bases of lo
plt.xlabel('Date')
plt.xticks(rotation=60)

plt.ylabel('Percent Changes')
plt.title('US Housing Price vs Stock, Bond and 10YT, GDP, UEMP, Housing data', fontsize=14)

plt.show()

#####################################
#Here 3/27/19

################################

from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

#Linear regression
from sklearn.linear_model import LinearRegression #import from sklearn
linear_reg= LinearRegression() # instantiated linreg
linear_reg.fit(X_train,y_train) #fit the model

#predict using X_test
predicted_train= linear_reg.predict(X_train)
predicted_test= linear_reg.predict(X_test)

from sklearn.metrics import mean_squared_error # import mse from sklearn
#calculate root mean squarred error
rmse_train=np.sqrt(mean_squared_error(y_train, predicted_train))
rmse_test=np.sqrt(mean_squared_error(y_test, predicted_test))

print('The train root mean squarred error is :', rmse_train)
print('The test root mean squarred error is :', rmse_test)

print('The Linear Regression coefficient parameters are :', linear_reg.coef_ )
print('The Linear Regression intercept value is :', linear_reg.intercept_)

from sklearn import metrics # import metrics from sklearn

Rsquared=linear_reg.score(X_train,y_train) # to determine r square Goodness of fit

# how good the model fits the training data can be determined by R squared metric which is here 0.12
Rsquared
print('The R squared metric is :', Rsquared)

'''The R^2 in scikit learn is the coefficient of determination. It is 1 - residual sum of square / total sum of squares.

'''

#####################################

'''
RMSE of the test data is closer to the training RMSE (and lower) if you have a well trained model. It will be higher if we have an overfitted model.
'''
import sklearn

from sklearn import metrics # import metrics from sklearn

Rsquared=linear_reg.score(X_train,y_train) # to determine r square Goodness of fit

# how good the model fits the training data can be determined by R squared metric which is here 0.12
Rsquared
print('The R squared metric is :', Rsquared)

'''The R^2 in scikit learn is the coefficient of determination. It is 1 - residual sum of square / total sum of squares.

Since R^2 = 1 - RSS/TSS, the only case where RSS/TSS > 1 happens when our model is even worse than the worst model assumed (which is the absolute mean model).

here RSS = sum of squares of difference between actual values(yi) and predicted values(yi^) and TSS = sum of squares of difference between actual values (yi) and mean value (Before applying Regression). So you can imagine TSS representing the best(actual) model, and RSS being in between our best model and the worst absolute mean model in which case we'll get RSS/TSS < 1. If our model is even worse than the worst mean model then in that case RSS > TSS(Since difference between actual observation and mean value < difference predicted value and actual observation).
'''
### K fold cross validation
# cross validation score
cv_score= cross_val_score(LinearRegression(),X,y,scoring='neg_mean_squared_error', cv=10) # k =10
print('cv_score is :', cv_score)

# mean squared error
print('cv_score is :', cv_score.mean())

# Root mean squared error
rmse_cv= np.sqrt(cv_score.mean() * -1)
print('The cross validation root mean squarred error is :', rmse_cv)

### with Linear regressor we are able to predict the model with .114 RMSE and r squared 0.33 and cross validation root mean squarred error is : 0.1507

########################################

'''Fitting Linear Regression using statsmodels

Statsmodels is a great Python library for a lot of basic and inferential
 statistics. It also provides basic regression functions using an R-like
 syntax, so it's commonly used by statisticians. While we don't cover
 statsmodels officially in the Data Science Intensive workshop,
 it's a good library to have in your toolbox. Here's a quick example
 of what you could do with it. The version of least-squares we will use
 in statsmodels is called ordinary least-squares (OLS). There are many
 other versions of least-squares such as partial least squares (PLS)
 and weighted least squares (WLS).
'''

#    http://www.statsmodels.org/devel/
housing_df.info()
# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import ols

# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m_H_RATIO_3A = ols('y ~ H_RATIO_3A_PCT_CHG',housing_df).fit()
print(m_H_RATIO_3A.summary())

m_HSN1F_3A = ols('y ~ HSN1F_3A_PCT_CHG',housing_df).fit()
print(m_HSN1F_3A.summary())

m_PERMIT_3A = ols('y ~ PERMIT_3A_PCT_CHG',housing_df).fit()
print(m_PERMIT_3A.summary())

m_STOCK_MKT_3A = ols('y ~ STOCK_MKT_3A_PCT_CHG',housing_df).fit()
print(m_STOCK_MKT_3A.summary())

m_BAA_YEILD_10Y_2A = ols('y ~ BAA_YEILD_10Y_2A_PCT_CHG',housing_df).fit()
print(m_BAA_YEILD_10Y_2A.summary())

m_US10Y_3A = ols('y ~ US10Y_3A_PCT_CHG',housing_df).fit()
print(m_US10Y_3A.summary())

m_RPCE_A = ols('y ~ RPCE_A_PCT_CHG',housing_df).fit()
print(m_RPCE_A.summary())

m_UEMP_3A = ols('y ~ UEMP_3A_PCT_CHG',housing_df).fit()
print(m_UEMP_3A.summary())

m_GDP_M = ols('y ~ RGDP_M_PCT_CHG',housing_df).fit()
print(m_GDP_M.summary())


m_rcpi = ols('y ~ H_RATIO_3A_PCT_CHG + HSN1F_3A_PCT_CHG + PERMIT_3A_PCT_CHG + STOCK_MKT_3A_PCT_CHG + BAA_YEILD_10Y_2A_PCT_CHG + US10Y_3A_PCT_CHG + RPCE_A_PCT_CHG + UEMP_3A_PCT_CHG + RGDP_M_PCT_CHG',housing_df).fit()
print(m_rcpi.summary())

fdval = m_rcpi.fittedvalues
y
plt.scatter(fdval, y)

#plt.xlim(3,8)
#plt.ylim(0,1)
plt.ylabel('Predicted prices')
plt.xlabel('Original Prices')
#plt.show()
#
#
sns.regplot(x=fdval, y="ASPUS_3A_PCT_CHG", data=housing_df, fit_reg = True, color='g')
plt.show()

###############################
'''
#Good
Correlation Matrix with Heatmap
Correlation states how the features are related to each other or the target variable.

Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)

Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.'''

import pandas as pd
import numpy as np
import seaborn as sns

X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

#get correlations of each features in dataset
corrmat = housing_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(12,12))
#plot heat map
g=sns.heatmap(housing_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

''' Let's Drop Fretures that are not important '''
    #target column
print(X[:-675,])  #print 1st 5 row of input
print(y[:-675,])

################################
#################  RAW DATA  ERROR TESTING  ##############################

h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df.csv', index_col='DATE', parse_dates=True)
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

#Linear regression
from sklearn.linear_model import LinearRegression #import from sklearn
linear_reg= LinearRegression() # instantiated linreg
linear_reg.fit(X_train,y_train) #fit the model

#predict using X_test
predicted_train= linear_reg.predict(X_train)
predicted_test= linear_reg.predict(X_test)

from sklearn.metrics import mean_squared_error # import mse from sklearn
#calculate root mean squarred error
rmse_train=np.sqrt(mean_squared_error(y_train, predicted_train))
rmse_test=np.sqrt(mean_squared_error(y_test, predicted_test))

print('The train root mean squarred error is :', rmse_train)
print('The test root mean squarred error is :', rmse_test)

print('The Linear Regression coefficient parameters are :', linear_reg.coef_ )
print('The Linear Regression intercept value is :', linear_reg.intercept_)

####### 

'''
RMSE of the test data is closer to the training RMSE (and lower) if you have a well trained model. It will be higher if we have an overfitted model.
'''

from sklearn import metrics # import metrics from sklearn

Rsquared=linear_reg.score(X_train,y_train) # to determine r square Goodness of fit

# how good the model fits the training data can be determined by R squared metric which is here 0.12
Rsquared
print('The R squared metric is :', Rsquared)

### K fold cross validation
# cross validation score
cv_score= cross_val_score(LinearRegression(),X,y,scoring='neg_mean_squared_error', cv=10) # k =10
print('cv_score is :', cv_score)

# mean squared error
print('cv_score is :', cv_score.mean())

# Root mean squared error
rmse_cv= np.sqrt(cv_score.mean() * -1)
print('The cross validation root mean squarred error is :', rmse_cv)

########################


############  RAW DATA FRAME  ########################

'''You will now apply what you have learned to display the aggregate mean values of each time series in the jobs DataFrame.


Extract the year for each of the dates in the index of jobs and assign them to index_year.
Compute the monthly mean unemployment rate in jobs and assign it to jobs_by_year.
Plot all the columns of price_by_year.
Show Answer (-42 XP)
Hint
To extract the year for each date, use the .index.year attribute.'''

# Extract of the year in each date indices of the jobs DataFrame
index_year = h_m_df.index.year

# Compute the mean unemployment rate for each year
h_m_df_by_year = h_m_df.groupby(index_year).mean()

# Plot the mean unemployment rate for each year
ax = h_m_df_by_year.plot(fontsize=12, linewidth=1.5)

# Set axis labels and legend
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('US Housing RAW Data', fontsize=10)
ax.legend(bbox_to_anchor=(0.1, 0.5), fontsize=10)
plt.yscale('symlog',basey=10)
plt.show()


index_year = housing_df.index.year

# Compute the mean unemployment rate for each year
housing_df_by_year = housing_df.groupby(index_year).mean()

# Plot the mean unemployment rate for each year
ax = housing_df_by_year.plot(fontsize=12, linewidth=1.5)

# Set axis labels and legend
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('US Housing RAW Data', fontsize=10)
ax.legend(bbox_to_anchor=(0.1, 0.5), fontsize=10)
plt.yscale('symlog',basey=10)
plt.show()

'''Averaging time series values by month shows that unemployment rate tends to be a lot higher during the winter months for the Agriculture and Construction industry. The increase in unemployment rate after 2008 is very clear when average time series values by year.'''


'''Apply time series decomposition to our dataset
We will now perform time series decomposition on multiple time series. We can achieve this by leveraging the Python dictionary to store the results of each time series decomposition.

Here, we will initialize an empty dictionary with a set of curly braces, {}, use a for loop to iterate through the columns of the DataFrame and apply time series decomposition to each time series. After each time series decomposition, we will place the results in the dictionary by using the command my_dict[key] = value, where my_dict is our dictionary, key is the name of the column/time series, and value is the decomposition object of that time series.


Initialize an empty dictionary called h_m_df_decomp.
Extract the column names of the housing DataFrame and place the results in a list called h_m_df_names.
Iterate through each column in h_m_df_names and apply time series decomposition to that time series. Place the results in the h_m_df_decomp dictionary, where the column name is the key, and the value is the decomposition of the time series just performed.

The columns of a DataFrame can be accessed by using the .columns attribute.
The basic structure of a for loop is:
for item in list:
    perform command'''

# Initialize dictionary
h_m_df_decomp = {}

# Get the names of each time series in the DataFrame
h_m_df_names = h_m_df.columns

# Run time series decomposition on each time series of the DataFrame
for ts in h_m_df_names:
    ts_decomposition = sm.tsa.seasonal_decompose(h_m_df[ts])
    h_m_df_decomp[ts] = ts_decomposition
#####
    
housing_df_decomp = {}

# Get the names of each time series in the DataFrame
housing_df_names = housing_df.columns

# Run time series decomposition on each time series of the DataFrame
for ts in housing_df_names:
    ts_decomposition = sm.tsa.seasonal_decompose(housing_df[ts])
    housing_df_decomp[ts] = ts_decomposition

'''Awesome! You've performed time series decomposition on all the time series in the jobs DataFrame. Let's try and plot them!'''


'''Visualize the seasonality of multiple time series
We will now extract the seasonality component of h_m_df_decomp to visualize the seasonality in these time series. Note that before plotting, you will have to convert the dictionary of seasonality components into a DataFrame using the pd.DataFrame.from_dict() function.

An empty dictionary h_m_df_seasonal and the time series decompisiton object h_m_df_decomp created.


Iterate through each column name in jobs_names and extract the corresponding seasonal component from h_m_df_decomp. Place the results in the jobs_seasonal, where the column name is the name of the time series, and the value is the seasonal component of the time series.
Convert h_m_df_seasonal to a DataFrame and call it seasonality_df.
Create a facetted plot of all 10 columns in seasonality_df. Ensure that the subgraphs do not share y-axis.

The seasonal component can be extracted using the .seasonal attribute.
Use the pd.DataFrame.from_dict() to convert a dictionary to a DataFrame.
Faceted plots of DataFrame df can be generated by setting the subplots argument to True.'''

# Extract the seasonal values for the decomposition of each time series
# Extract the seasonal values for the decomposition of each time series
h_m_df_seasonal = {}

for ts in h_m_df_names:
    h_m_df_seasonal[ts] = h_m_df_decomp[ts].seasonal

# Create a DataFrame from the housing_seasonal dictionnary
seasonality_df = pd.DataFrame.from_dict(h_m_df_seasonal)

# Remove the label for the index
seasonality_df.index.name = None

# Create a faceted plot of the seasonality_df DataFrame
seasonality_df.plot(subplots=True,
                   layout=(5, 2),
                   sharey=False,
                   fontsize=6,
                   linewidth=0.9,
                   legend=True)

# Show plot
plt.show()

'''Wow! Each time series in the h_m_df DataFrame have very different seasonality patterns!'''

'''Correlations between multiple time series
Earlier, we have extracted the seasonal component of each time series in the h_m_df DataFrame and stored those results in new DataFrame called seasonality_df. In the context of h_m_df data, it can be interesting to compare seasonality behavior, as this may help uncover which h_m_df indicators are the most similar or the most different.

This can be achieved by using the seasonality_df DataFrame and computing the correlation between each time series in the dataset. Here, we will compute and create a clustermap visualization of the correlations between time series in the seasonality_df DataFrame.

Compute the correlation between all columns in the seasonality_df DataFrame using the spearman method and assign the results to seasonality_corr.
Create a new clustermap of your correlation matrix.

Use the .corr() method along with the method argument to create a correlation matrix.
To plot a clustermap, use the sns.clustermap() function.
'''

# Get correlation matrix of the seasonality_df DataFrame
seasonality_corr = seasonality_df.corr(method='spearman')

# Customize the clustermap of the seasonality_corr correlation matrix
fig = sns.clustermap(seasonality_corr, annot=True, annot_kws={"size": 12}, linewidths=.8, figsize=(15, 10))
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.show()
#######

housing_df_seasonal = {}

for ts in housing_df_names:
    housing_df_seasonal[ts] = housing_df_decomp[ts].seasonal

# Create a DataFrame from the jobs_seasonal dictionnary
seasonality_df = pd.DataFrame.from_dict(housing_df_seasonal)

# Remove the label for the index
seasonality_df.index.name = None

# Create a faceted plot of the seasonality_df DataFrame
seasonality_df.plot(subplots=True,
                   layout=(5, 2),
                   sharey=False,
                   fontsize=6,
                   linewidth=0.9,
                   legend=True)

# Show plot
plt.show()


'''Wow! Each time series in the jobs DataFrame have very different seasonality patterns!'''

'''Correlations between multiple time series
Earlier, we have extracted the seasonal component of each time series in the h_m_df DataFrame and stored those results in new DataFrame called seasonality_df. In the context of h_m_df data, it can be interesting to compare seasonality behavior, as this may help uncover which h_m_df indicators are the most similar or the most different.

This can be achieved by using the seasonality_df DataFrame and computing the correlation between each time series in the dataset. Here, we will compute and create a clustermap visualization of the correlations between time series in the seasonality_df DataFrame.

Compute the correlation between all columns in the seasonality_df DataFrame using the spearman method and assign the results to seasonality_corr.
Create a new clustermap of your correlation matrix.

Use the .corr() method along with the method argument to create a correlation matrix.
To plot a clustermap, use the sns.clustermap() function.
'''

# Get correlation matrix of the seasonality_df DataFrame
seasonality_corr = seasonality_df.corr(method='spearman')

# Customize the clustermap of the seasonality_corr correlation matrix
fig = sns.clustermap(seasonality_corr, annot=True, annot_kws={"size": 12}, linewidths=.8, figsize=(15, 10))
plt.setp(fig.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(fig.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
plt.show()

################  RAW Data ###################  ML Lin Reg

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Prepare input and output DataFrames
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# Fit the model
model = linear_model.LinearRegression()
model.fit(X, y)
        #H_RATIO_M        HSN1F_M  		    PERMIT_M       STOCK_MKT_M 		   BAA10YM 	       US10Y_M       	  RPCE_M  		  LRUN_UEMP         GDP_M
new_inputs = np.array([[ 6.60000000e+00,  6.21000000e+02,  1.32600000e+03,  -5.88567149e+00,  2.30000000e+00,  2.83000000e+00,   2.80000000e+00,  3.62318061e+00,  2.60000000e+00],
       [ 4.50000000e+00,  3.85000000e+02,  9.30000000e+02,   3.10174247e+00,  3.12000000e+00,  1.72000000e+00,   1.43000000e+00,  7.79994189e+00,  5.00000000e-01],
       [ 6.30000000e+00,  1.07400000e+03,  1.86700000e+03,   9.36840249e-01,  1.67000000e+00,  5.11000000e+00,   2.27000000e+00,  4.69406217e+00,  7.00000000e-01],
       [ 4.30000000e+00,  9.00000000e+02,  1.65100000e+03,   2.46476236e+00,  2.11000000e+00,  6.26000000e+00,   4.70000000e+00,  4.08157421e+00,  5.50000000e+00],
       [ 4.50000000e+00,  8.12000000e+02,  1.46100000e+03,   1.61322058e+00,  1.92000000e+00,  5.77000000e+00,   4.33000000e+00,  6.91357601e+00,  4.46666667e+00],
       [ 6.80000000e+00,  6.66000000e+02,  1.51400000e+03,  -1.06550895e+01,  1.89000000e+00,  9.42000000e+00,   2.13000000e+00,  5.76349116e+00,  5.83333333e+00],
       [ 9.20000000e+00,  4.15000000e+02,  9.76000000e+02,  -2.83633570e+00,  2.33000000e+00,  1.34700000e+01,   1.27000000e+00,  7.40560669e+00,  2.30000000e+00],
       [ 8.90000000e+00,  4.77000000e+02,  7.09000000e+02,   1.34337920e+01,  2.75000000e+00,  7.73000000e+00,   5.60000000e+00,  8.90922651e+00,  3.33333333e-01],
       [ 5.20000000e+00,  5.11000000e+02,  1.38900000e+03,  -7.64093873e-01,  1.20000000e+00,  6.03000000e+00,   3.60000000e+00,  3.49405492e+00,  4.80000000e+00],
       [ 4.70000000e+00,  5.91000000e+02,  1.22300000e+03,  6.97363277e-01,  1.05000000e+00,   3.98000000e+00,   4.93000000e+00,  5.12848311e+00,  2.53333333e+00]])

type(new_inputs)
new_inputs.shape
z=new_inputs.reshape(1, -1)
print(z)
type(z)
z.shape
'''array([
378900.     ,
296633.3333 ,
306266.6667 ,
202566.6667 ,
151833.3333 ,
132300.     ,
83966.66667,
42033.33333,
27400.     ,
19300.     ])'''
    

new_inputs1 = np.array([[ 8.60000000e+00,  2.21000000e+02,  1.32600000e+03,  -7.88567149e+00,  2.30000000e+00,  3.83000000e+00,   3.80000000e+00,  4.62318061e+00,  3.60000000e+00],
       [ 2.50000000e+00,  6.85000000e+02,  19.30000000e+02,   13.10174247e+00,  7.12000000e+00,  0.72000000e+00,   4.43000000e+00,  2.79994189e+00,  9.00000000e-01],
       [12.30000000e+00,  12.07400000e+03,  11.86700000e+03,  -9.36840249e-01,  0.67000000e+00,  1.11000000e+00,   1.27000000e+00,  14.69406217e+00,  2.00000000e-01],
       [ 0.30000000e+00,  9.00000000e+02,  1.65100000e+03,   12.46476236e+00,  2.11000000e+00,  1.26000000e+00,   4.70000000e+00,  4.08157421e+00,  5.50000000e+00],
       [ 4.50000000e+00,  8.12000000e+02,  1.46100000e+03,   1.61322058e+00,  1.92000000e+00,  5.77000000e+00,   4.33000000e+00,  6.91357601e+00,  4.46666667e+00],
       [ 16.80000000e+00,  6.66000000e+02,  1.51400000e+03,  -1.06550895e+01,  1.89000000e+00,  9.42000000e+00,   2.13000000e+00,  5.76349116e+00,  5.83333333e+00],
       [ 6.20000000e+00,  2.15000000e+02,  9.76000000e+02,  2.83633570e+00,  2.33000000e+00,  2.34700000e+00,   1.27000000e+00,  3.40560669e+00,  4.30000000e+00],
       [ 18.90000000e+00,  4.77000000e+02,  7.09000000e+02,   1.34337920e+01,  2.75000000e+00,  7.73000000e+00,   5.60000000e+00,  8.90922651e+00,  3.33333333e-01],
       [5.20000000e+00,  5.11000000e+02,  1.38900000e+03,  -7.64093873e-01,  1.20000000e+00,  6.03000000e+00,   3.60000000e+00,  3.49405492e+00,  4.80000000e+00],
       [ 6.70000000e+00,  1.91000000e+02,  1.22300000e+03,  -6.97363277e-01,  1.05000000e+00,   4.98000000e+00,   3.93000000e+00,  4.12848311e+00,  2.53333333e+00]])


#.reshape(1, -1)

predictions = model.predict(new_inputs)
print(predictions)

predictions = model.predict(new_inputs1)
print(predictions)
                        #H_RATIO_M        HSN1F_M  		    PERMIT_M       STOCK_MKT_M 		   BAA10YM 	       US10Y_M       	  RPCE_M  		  LRUN_UEMP         GDP_M
new_inputs2 = np.array([ 6.60000000e+00,  6.21000000e+02,  1.32600000e+03,  -5.88567149e+00,  2.30000000e+00,  2.83000000e+00,   2.80000000e+00,  3.62318061e+00,  2.60000000e+00])
predictions = model.predict(new_inputs2.reshape(1, -1))
print(predictions)

new_inputs_up = np.array([ 9.60000000e+00,  9.21000000e+02,  0.32600000e+03,  5.88567149e+00,  3.30000000e+00,  1.83000000e+00,   1.80000000e+00,  1.62318061e+00,  4.60000000e+00])
predictions = model.predict(new_inputs_up.reshape(1, -1))
print(predictions)

new_inputs_down = np.array([ 5.60000000e+00,  5.21000000e+02,  2.32600000e+03,  -6.88567149e+00,  1.80000000e+00,  3.23000000e+00,   3.30000000e+00,  4.62318061e+00,  1.60000000e+00])
predictions = model.predict(new_inputs_down.reshape(1, -1))
print(predictions)

# Visualize the inputs and predicted values
plt.scatter([new_inputs_down], predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()

predictions = model.predict(new_inputs1)
print(predictions)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_1d = pca.fit_transform(new_inputs1)
x_1d.ravel()
plt.scatter(x_1d.ravel(), predictions, color='r', s=18)
#y_pred, y_fit,
plt.plot(x_1d.ravel(), predictions, color='b', linewidth=2, label = 'Linear regression\n'+reg_label)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()



plt.scatter(new_inputs1.reshape(1, -1), predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()


# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=7)
print(scores)



'''Yes! As you can see, fitting a model with raw data doesn't give great results.'''

'''Visualizing predicted values
When dealing with time series data, it's useful to visualize model predictions on top of the "actual" values that are used to test the model.

In this exercise, after splitting the data (stored in the variables X and y) into training and test sets, you'll build a model and then visualize the model's predictions on top of the testing data in order to estimate the model's performance.

Split the data (X and y) into training and test sets.
Use the training data to train the regression model.
Then use the testing data to generate predictions for the model.

You should be splitting up the arrays X and y into training and test sets with train_test_split().

Coefficient of Determination (R^2 )
The value of R is bounded on the top by 1, and can be infinitely low
Values closer to 1 mean the model does a better job of predicting outputs'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing_df.info()
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

# Split our data into training and test sets
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.295, random_state=42)
#
# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)



##########
print('r2_score for h_m_s_df (Transformed RAW Hosing Data) :')
X = np.array(h_m_s_df.drop(['ASPUS_M'],1))
y = np.array(h_m_s_df['ASPUS_M'])

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.297, random_state=42)
#
# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)




'''Plot a time series of the predicted and "actual" values of the testing data. The predicted values are stored in predictions.'''
'''
#Check Data Types
print(type(y_test))
print(type(predictions))

#Lets convert prediction to dataframe
predictions = pd.DataFrame(predictions)
predictions.index = y_test.index

all_prices.head()
predictions.head()
y_test.head()

#fig, ax = plt.subplots(figsize=(15, 5))
ax = y_test.plot(color='b', lw=2)
predictions.plot(color='r', lw=1, ax=ax)

#plt.xlim(all_prices.index.min(), all_prices.index.max())
#plt.ylim(all_prices.price.min(), all_prices.price.max())
plt.tight_layout()

# plt.savefig('test_mline.png', dpi=150)
plt.show()
X_predict = h_m_df[['GDP_M', 'US10Y_M']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)
# Generate predictions with the model using those inputs
predictions = model.predict(X)
# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()'''

'''Good job! Here the red line shows the relationship that your model found. As the proportion of pre-1940s houses gets larger, the average number of rooms gets slightly lower.'''


'''Good job! Here the red line shows the relationship that your model found. As the proportion of pre-1940s houses gets larger, the average number of rooms gets slightly lower.'''

X[1:]
################
'''                    PCA
use PCA to decorrelate these measurements, then plot the decorrelated points and measure their Pearson correlation.

'''
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0, 1]

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(housing_df)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()



'''
Variance of the PCA features

The dataset is 10-dimensional. But what is its intrinsic dimension? Make a plot of the variances of the PCA features to find out. As before, samples is a 2D array, where each row represents a fish. You'll need to standardize the features first.

the use of principal component analysis for dimensionality reduction, for visualization of high-dimensional data, for noise filtering, and for feature selection within high-dimensional data. Because of the versatility and interpretability of PCA, it has been shown to be effective in a wide variety of contexts and disciplines. Given any high-dimensional dataset, I tend to start with PCA in order to visualize the relationship between points ), to understand the main variance in the data and to understand the intrinsic dimensionality (by plotting the explained variance ratio).
'''
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(housing_df)
pca.n_components_

print(housing_df.columns)
#df_pca = pd.read_csv(url, names=['Loan Amount', 'zip_code', 'loan_purpose_code', 'Qualification FICO', 'unit_type_code', 'loan_type_code', 'Fix_True', 'CLTV', 'RATE', 'Home'])
# Plot the explained variances
columns = ['ASPUS_3A_PCT_CHG', 'H_RATIO_3A_PCT_CHG', 'HSN1F_3A_PCT_CHG',
       'PERMIT_3A_PCT_CHG', 'STOCK_MKT_3A_PCT_CHG', 'BAA_YEILD_10Y_2A_PCT_CHG',
       'US10Y_3A_PCT_CHG', 'RPCE_A_PCT_CHG', 'UEMP_3A_PCT_CHG', 'GDP_M']

ax = housing_df.plot( fontsize=6)
for x in ax.get_xticklabels(minor=False):
    #columns[::100]
    print(x)

ax.set_xticks(np.arange(len(housing_df.index)))
ax.set_xticklabels([case for case in housing_df.columns], rotation=30)

'''Now we want to know how many principal components we can choose for our new feature subspace? A useful measure is the so-called “explained variance ratio“. The explained variance ratio tells us how much information (variance) can be attributed to each of the principal components. We can plot bar graph between no. of features on X axis and variance ratio on Y axis'''

features = range(pca.n_components_)
feature_names = features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(feature_names)
plt.show()

plt.plot([1, 9])
#plt.plot(['ASPUS_3A_PCT_CHG', 'H_RATIO_3A_PCT_CHG', 'HSN1F_3A_PCT_CHG',       'PERMIT_3A_PCT_CHG', 'STOCK_MKT_3A_PCT_CHG', 'BAA_YEILD_10Y_2A_PCT_CHG',       'US10Y_3A_PCT_CHG', 'RPCE_A_PCT_CHG', 'UEMP_3A_PCT_CHG', 'GDP_M'])
ax = plt.gca()
labels = ax.get_xticklabels()
for label in labels:
    print(label)

pca.fit_transform(X)
print(pca.mean_)
print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.n_components_)
print(pca.noise_variance_)


'''Finding Correlation between Features and Target Variable in housing Dataset using Heatmap'''

correlation = housing_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation')

columns

####
#Let us load the basic packages needed for the PCA analysis

pca = PCA().fit(housing_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()


'''
To see what these numbers mean, let's visualize them as vectors over the input data, using the "components" to define the direction of the vector, and the "explained variance" to define the squared-length of the vector:'''

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 4], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

'''
PCA as dimensionality reduction
Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance.

Here is an example of using PCA as a dimensionality reduction transform:'''

pca = PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

'''
The transformed data has been reduced to a 6 dimension. To understand the effect of this dimensionality reduction, we can perform the inverse transform of this reduced data and plot it along with the original data:'''

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 4], alpha=0.7, c='red')
plt.scatter(X_new[:, 0], X_new[:, 5], alpha=0.5, c='blue')
plt.axis('equal');
#ylim(-20, 100)
#xlim(0,1800000)
plt.figure(figsize=(16,16))
plt.show()



################################

'''Fitting Linear Regression using sklearn'''
from sklearn.linear_model import LinearRegression

# This creates a LinearRegression object
lm = LinearRegression()

'''
lm.predit()
lm.fit()
lm.score()
lm.coef_
lm.intercept_

Fit a linear model
The lm.fit() function estimates the coefficients the linear regression using least squares.
'''
# Look inside lm object
dir(lm)

# Use all 13 predictors to fit linear regression model
lm.fit(X, y)
lm.coef_

lm.intercept_
housing_df.info()
# The mean squared error
print("Mean squared error (ASPUS_3A_PCT_CHG): %.2f" % np.mean((lm.predict(X) - y) ** 2))

################################

housing_df= pd.read_csv('C:/scripts/capstone2/housing_df.csv', index_col=0)
housing_df.head()

housing_subset_1 = housing_df['1/1/1979':'1/1/1982']
ax = housing_subset_1.plot(fontsize=15)

# Plot the time series in your DataFrame as a blue area chart


'''Subset time series data
When plotting time series data, you may occasionally want to visualize only a subset of the data. The pandas library provides powerful indexing and subsetting methods that allow you to extract specific portions of a DataFrame. For example, you can subset all the data between 1950 and 1960 in the discoveries DataFrame by specifying the following date range:

subset_data = discoveries['1950-01-01':'1960-01-01']
Note: Subsetting your data this way is only possible if the index of your DataFrame contains dates of the datetime type. Failing that, the pandas library will return an error message.

Use discoveries to create a new DataFrame discoveries_subset_2 that contains all the data between January 1, 1939 and January 1, 1958.
Plot the time series of discoveries_subset_2 using a "blue" line plot.

To specify a date range, use the format YYYY-MM-DD: YYYY-MM-DD.'''


# Select the subset of data between 1939 and 1958
housing_subset_2 = housing_df['2007-01-01':'2010-01-01']

# Plot the time series in your DataFrame as a blue area chart
ax = housing_subset_2.plot(fontsize=15)



housing_subset_3 = housing_df['2018-01-01':'2018-12-01']

# Plot the time series in your DataFrame as a blue area chart
ax = housing_subset_3.plot(fontsize=15)

# Show plot
plt.show()

# Plot your the discoveries time series
ax = housing_df.plot( fontsize=6)

# Add a red vertical line
ax.axvline('2007-01-01', color='red', linestyle='--')
ax.axvline('2009-01-01', color='red', linestyle='--')


# Add a green horizontal line
ax.axhline(4, color='green', linestyle='--')
ax.axhline(-4, color='green', linestyle='--')

ax = housing_df.plot(fontsize=6)
plt.show()
# Add a vertical red shaded region between the dates of 1900-01-01 and 1915-01-01
ax.axvspan('2007-9-01', '2008-10-01', color='red', alpha=0.2)

# Add a horizontal green shaded region between the values of 6 and 8
ax.axhspan(-3, -4, color='green', alpha=0.2)

plt.show()

##########

# Select the subset of data between 2007 and 2010
housing_subset_2 = housing_df['1/1/2007':'1/1/2010']

# Plot the time series in your DataFrame as a blue area chart
ax = housing_subset_2.plot(fontsize=15)

housing_subset_3 = housing_df['1/1/2018':'12/1/2018']

# Plot the time series in your DataFrame as a blue area chart
ax = housing_subset_3.plot(fontsize=15)

# Show plot
plt.show()

# Plot your the discoveries time series
ax = housing_df.plot( fontsize=6)

# Add a red vertical line
ax.axvline('1/1/2007', color='red', linestyle='--')
ax.axvline('1/1/2009', color='red', linestyle='--')


# Add a green horizontal line
ax.axhline(4, color='green', linestyle='--')
ax.axhline(-4, color='green', linestyle='--')

ax = housing_df.plot(fontsize=6)
plt.show()
# Add a vertical red shaded region between the dates of 2007-01-01 and 2008-101-01
ax.axvspan('9/1/2007', '10/12008', color='red', alpha=0.2)

# Add a horizontal green shaded region between the values of 6 and 8
ax.axhspan(-3, -4, color='green', alpha=0.2)

plt.show()

################################

'''Display rolling averages
It is also possible to visualize rolling averages of the values in your time series. This is equivalent to "smoothing" your data, and can be particularly useful when your time series contains a lot of noise or outliers. For a given DataFrame df, you can obtain the rolling average of the time series by using the command:

df_mean = df.rolling(window=12).mean()
The window parameter should be set according to the granularity of your time series. For example, if your time series contains daily data and you are looking for rolling values over a whole year, you should specify the parameter to window=365. In addition, it is easy to get rolling values for other other metrics, such as the standard deviation (.std()) or variance (.var()).

Compute the 52 weeks rolling mean of co2_levels and assign it to ma.
Compute the 52 weeks rolling standard deviation of co2_levels and assign it to mstd.
Calculate the upper bound of time series which can defined as the rolling mean + (2 * rolling standard deviation) and assign it to ma[upper]. Similarly, calculate the lower bound as the rolling mean - (2 * rolling standard deviation) and assign it to ma[lower].
Plot the line chart of ma.

The rolling metric of DataFrame df can be computed by using the command df.rolling(window=time_period).your_metric().
You can use the .mean() and the .std() methods to calculate the mean and the standard deviation.'''
housing_df.info()
# Compute the 52 weeks rolling mean of the co2_levels DataFrame
ma = housing_df.rolling(window=26).mean()

# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame
mstd = housing_df.rolling(window=26).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['ASPUS_3A_PCT_CHG'] + (mstd['ASPUS_3A_PCT_CHG'] * 2)

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['ASPUS_3A_PCT_CHG'] - (mstd['ASPUS_3A_PCT_CHG'] * 2)

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=2.8, fontsize=12)

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('ASPUS_3A_PCT_CHG', fontsize=12)
ax.set_title('Rolling mean (26 Weeks) and variance of AVE Housing Price levels\nin USA from 1962 to 2018', fontsize=12)
plt.show()


ma = housing_df.rolling(window=52).mean()

# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame
mstd = housing_df.rolling(window=52).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['ASPUS_3A_PCT_CHG'] + (mstd['ASPUS_3A_PCT_CHG'] * 2)

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['ASPUS_3A_PCT_CHG'] - (mstd['ASPUS_3A_PCT_CHG'] * 2)

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=2.8, fontsize=12)

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('ASPUS_3A_PCT_CHG', fontsize=12)
ax.set_title('Rolling mean (52 Weeks) and variance of AVE Housing Price levels\nin USA from 1962 to 2018', fontsize=12)
plt.show()

''' Showing the rolling mean and standard deviation of your data allows to get a more compact view of your data.'''
################################


'''Display aggregated values
You may sometimes be required to display your data in a more aggregated form. For example, the co2_levels data contains weekly data, but you may need to display its values aggregated by month of year. In datasets such as the co2_levels DataFrame where the index is a datetime type, you can extract the year of each dates in the index:

# extract of the year in each dates of the df DataFrame
index_year = df.index.year
To extract the month or day of the dates in the indices of the df DataFrame, you would use df.index.month and df.index.day, respectively. You can then use the extracted year of each indices in the co2_levels DataFrame and the groupby function to compute the mean CO2 levels by year:

df_by_year = df.groupby(index_year).mean()
Instructions
70 XP
Extract the month for each of the dates in the index of the co2_levels DataFrame and assign the values to a variable called index_month.
Using the groupby and mean functions from the pandas library, compute the monthly mean CO2 levels in the co2_levels DataFrame and assign that to a new DataFrame called mean_co2_levels_by_month.
Plot the values of the mean_co2_levels_by_month DataFrame using a fontsize of 6 for the axis ticks.

The month values of the indices in a DataFrame df can be extracted using the command df.index.month.
Assuming you have successfully created the index_month variable, the monthly sum a DataFrame df can be computed using the command df.groupby(index_month).sum().
The mean can be computed by replacing the term sum() with mean().'''

# Get month for each dates in the index of co2_levels
index_month = ASPUS_3A.index.month

# Compute the mean ASPUS_3A for each month of the year
mean_ASPUS_3A_by_month = ASPUS_3A.groupby(index_month).mean()

# Plot the mean ASPUS_3A for each month of the year
mean_ASPUS_3A_by_month.plot(fontsize=6)

# Specify the fontsize on the legend
plt.legend(fontsize=10)

# Show plot
plt.show()


# Get year for each dates in the index of ASPUS_3A
index_year = ASPUS_3A.index.year

# Compute the mean ASPUS_3A for each month of the year
mean_ASPUS_3A_by_year = ASPUS_3A.groupby(index_year).mean()

# Plot the mean ASPUS_3A for each month of the year
mean_ASPUS_3A_by_year.plot(fontsize=6)

# Specify the fontsize on the legend
plt.legend(fontsize=10)

# Show plot
plt.show()


#################################

# Generate a boxplot
ax = ASPUS_3A.boxplot()

# Set the labels and display the plot
ax.set_xlabel('ASPUS_3A', fontsize=10)
ax.set_ylabel('Boxplot ASPUS_3A in USA', fontsize=10)
plt.legend(fontsize=10)
print(ASPUS_3A.quantile([.10, 0.25, .50, .75, 0.90, .999]))
plt.title('This box plot display property growth in 3 years')
plt.show()

'''Use the .plot() method with kind = 'hist' along with the bins argument.'''


# Generate a histogram
ax = ASPUS_3A.plot(kind='hist', bins=50, fontsize=6)

# Set the labels and display the plot
ax.set_xlabel('ASPUS_3A', fontsize=10)
ax.set_ylabel('Histogram of ASPUS_3A levels in USA', fontsize=10)
plt.legend(fontsize=10)
plt.show()

''' Hopefully, this shows how boxplots can be a good graphical alternative to numerical summaries.'''

##################################

'''Density plots
In practice, histograms can be a substandard method for assessing the distribution of data because they can be strongly affected by the number of bins that have been specified. Instead, kernel density plots represent a more effective way to view the distribution of your data. An example of how to generate a density plot of is shown below:

ax = df.plot(kind='density', linewidth=2)
The standard .plot() method is specified with the kind argument set to 'density'. We also specified an additional parameter linewidth, which controls the width of the line to be plotted.

Using the ASPUS_3A DataFrame, produce a density plot of the ASPUS_3A data with line width parameter of 4.
Annotate the x-axis labels of your boxplot with the string 'ASPUS_3A'.
Annotate the y-axis labels of your boxplot with the string 'Density plot of ASPUS_3A levels in USA'.

Use the .plot() method with kind = 'density' along with the linewidth argument.
The x and y labels can be set using the .set_xlabel() and .set_ylabel() methods.'''


# Display density plot of CO2 levels values
ax = ASPUS_3A.plot(kind='density', linewidth=4, fontsize=6)

# Annotate x-axis labels
ax.set_xlabel('ASPUS_3A', fontsize=10)

# Annotate y-axis labels
ax.set_ylabel('Density plot of ASPUS_3A in USA', fontsize=10)

plt.show()

#################################
#Time Series

'''Autocorrelation in time series data
In the field of time series analysis, autocorrelation refers to the correlation of a time series with a lagged version of itself. For example, an autocorrelation of order 3 returns the correlation between a time series and its own values lagged by 3 time points.

It is common to use the autocorrelation (ACF) plot, also known as self-autocorrelation, to visualize the autocorrelation of a time-series. The plot_acf() function in the statsmodels library can be used to measure and plot the autocorrelation of a time series.


Import tsaplots from statsmodels.graphics.
Use the plot_acf() function from tsaplots to plot the autocorrelation of the 'ASPUS_3A' column in ASPUS_3A levels.
Specify a maximum lag of 36.

To use plot_acf() from tsaplots, use the command tsaplots.plot_acf().'''

# Import required libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.graphics import tsaplots

# Display the autocorrelation plot of your time series
fig = tsaplots.plot_acf(ASPUS_3A['ASPUS_3A_PCT_CHG'], lags=36)

# Show plot
plt.show()

''' Autocorrelation plots can be used to quickly discover patterns into your 
time series, so let's delve a little bit deeper into that!'''

'''Partial autocorrelation in time series data
Like autocorrelation, the partial autocorrelation function (PACF) measures 
the correlation coefficient between a time-series and lagged versions of 
itself. However, it extends upon this idea by also removing the effect of 
previous time points. For example, a partial autocorrelation function of 
order 3 returns the correlation between our time series (t_1, t_2, t_3, ...) 
and its own values lagged by 3 time points (t_4, t_5, t_6, ...), but only 
after removing all effects attributable to lags 1 and 2.

The plot_pacf() function in the statsmodels library can be used to measure and plot the partial autocorrelation of a time series.


Import tsaplots from statsmodels.graphics.
Use the plot_pacf() function from tsaplots to plot the partial autocorrelation of the 'ASPUS_3A_PCT_CHG' column in ASPUS_3A.
Specify a maximum lag of 36.

To use plot_pacf() from tsaplots, use the command tsaplots.plot_pacf().'''

# Import required libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.graphics import tsaplots

# Display the autocorrelation plot of your time series
fig = tsaplots.plot_pacf(ASPUS_3A['ASPUS_3A_PCT_CHG'], lags=36)

# Show plot
plt.show()

'''Just like autocorrelation, partial autocorrelation plots can be tricky to interpret, so let's test analyze those!'''

###########################################

'''Interpret partial autocorrelation plots
If partial autocorrelation values are close to 0, then values between observations and lagged observations are not correlated with one another. Inversely, partial autocorrelations with values close to 1 or -1 indicate that there exists strong positive or negative correlations between the lagged observations of the time series.

The .plot_pacf() function also returns confidence intervals, which are represented as blue shaded regions. If partial autocorrelation values are beyond this confidence interval regions, then you can assume that the observed partial autocorrelation values are statistically significant.

In the partial autocorrelation plot below, at which lag values do we have statistically significant partial autocorrelations?


Check the values of the partial autocorrelations and whether they are beyond the confidence interval regions.'''
###############################################


'''Time series decomposition
When visualizing time series data, we should look out for some distinguishable patterns:

seasonality: does the data display a clear periodic pattern?
trend: does the data follow a consistent upwards or downward slope?
noise: are there any outlier points or missing values that are not consistent with the rest of the data?
You can rely on a method known as time-series decomposition to automatically extract and quantify the structure of time-series data. The statsmodels library provides the seasonal_decompose() function to perform time series decomposition out of the box.

decomposition = sm.tsa.seasonal_decompose(time_series)
You can extract a specific component, for example seasonality, by accessing the seasonal attribute of the decomposition object.

Import statsmodels.api using the alias sm.
Perform time series decomposition on the ASPUS_3A DataFrame into a variable called decomposition.
Print the seasonality component of your time series decomposition.

Use the seasonal_decompose() function to perform time series decompisiton.
To print the seasonality component, access the seasonal component of the decomposition object.'''

#GOOD
#datestamp  import DF without index
ASPUS2 = pd.read_csv('C:/scripts/capstone2/ASPUS.csv')

type(ASPUS2)
# Display first seven rows of co2_levels
print(ASPUS2.head(7))

# Convert the date column to a datestamp type
ASPUS2['DATE'] = pd.to_datetime(ASPUS2['DATE'])

# Set datestamp column as index
ASPUS2 = ASPUS2.set_index('DATE')

# Print out the number of missing values
print(ASPUS2.isnull().sum())

ASPUS2.info()

# Impute missing values with the next valid observation
ASPUS2 = ASPUS2.interpolate(method='linear')
ASPUS2 = ASPUS2.fillna(method='bfill')

# Print out the number of missing values
print(ASPUS2.isnull().sum())
ASPUS2.info()

# Import statsmodels.api as sm
import statsmodels.api as sm

# Perform time series decompositon
decomposition = sm.tsa.seasonal_decompose(ASPUS)

# Print the seasonality component
print(decomposition.seasonal)

'''Excellent! Time series decomposition is a powerful method to reveal the structure of our time series. Now let's visualize these components.'''

'''Plot individual components
It is also possible to extract other inferred quantities from our time-series decomposition object. The following code shows you how to extract the observed, trend and noise (or residual, resid) components.

observed = decomposition.observed
trend = decomposition.trend
residuals = decomposition.resid
We can then use the extracted components and plot them individually.

The decomposition object we have created already.


Extract the trend component from the decomposition object.
Plot this trend component.

Individual components can be extracted by accessing the relevant attribute.'''

# Extract the trend component
trend = decomposition.trend
observed = decomposition.observed
residuals = decomposition.resid
seasonal = decomposition.seasonal

# Plot the values of the trend
ax = trend.plot(figsize=(12, 6), fontsize=12)

# Specify axis labels
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Seasonal component the ASPUS time-series\nAverage Sales Price of Houses Sold for the United States', fontsize=12)
plt.yscale('symlog',basey=1000000)
plt.show()

ax = observed.plot(figsize=(12, 6), fontsize=8)

# Specify axis labels
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Observed component the ASPUS time-series\nAverage Sales Price of Houses Sold for the United States', fontsize=10)
plt.yscale('symlog',basey=1000000)
plt.show()

ax = residuals.plot(figsize=(12, 6), fontsize=8)
# Specify axis labels
ax.set_xlabel('Date', fontsize=12)
ax.set_title('residuals component the ASPUS time-series\nAverage Sales Price of Houses Sold for the United States', fontsize=10)
#plt.yscale('symlog',basey=10)
plt.show()

ax = seasonal.plot(figsize=(12, 6), fontsize=8)
# Specify axis labels
ax.set_xlabel('Date', fontsize=12)
ax.set_title('seasonal component the ASPUS time-series\nAverage Sales Price of Houses Sold for the United States', fontsize=10)
#plt.yscale('symlog',basey=200000)
plt.show()
############################
'''We placed the trend and seasonal components in the airline_decomposed DataFrame.

Print the first 5 rows of airline_decomposed.
Plot these two components on the same graph.
Hint
Use the .head(5) method to print the first 5 rows.
Call .plot() on airline_decomposed.'''

#GOOD

#JOIN 2 Data frame on DateTimeIndex outer join
trend = trend.interpolate(method='linear')
trend = trend.fillna(method='bfill')
trend = trend.fillna(method='ffill')
trend = trend[['ASPUS_M']]
seasonal = seasonal.interpolate(method='linear')
seasonal = seasonal.fillna(method='bfill')
seasonal = seasonal.fillna(method='ffill')
seasonal = seasonal[['ASPUS_M']]
# Print out the number of missing values
print(seasonal.isnull().sum())
print(trend.isnull().sum())
ASPUS2.info()
ASPUS_decomposed= trend.merge(seasonal, how='outer', left_index=True, right_index=True)
ASPUS = ASPUS.interpolate(method='linear')
ASPUS = ASPUS.fillna(method='bfill')

# Print out the number of missing values
print(ASPUS.isnull().sum())
ASPUS.info()
ASPUS_decomposed.columns = ['trend', 'seasonal']
ASPUS_decomposed.info()

type(ASPUS_decomposed)

# Print the first 5 rows of ASPUS_decomposed
print(ASPUS_decomposed.tail(5))

# Plot the values of the df_decomposed DataFrame
ax = ASPUS_decomposed.plot(figsize=(12, 6), fontsize=15)

# Specify axis labels
ax.set_xlabel('Date', fontsize=15)
plt.legend(fontsize=15)
plt.yscale('symlog',basey=10)
plt.show()
############################

'''Visualize the Housing Market dataset
You will now review the contents of chapter 1. You will have the opportunity to work with a new dataset that contains the monthly number of passengers who took a commercial flight between January 1949 and December 1960.

We have printed the first 5 and the last 5 rows of the airline DataFrame for you to review.

Plot the time series of airline using a "blue" line plot.
Add a vertical line on this plot at December 1, 1955.
Specify the x-axis label on your plot: 'Date'.
Specify the title of your plot: 'Number of Monthly Airline Passengers'.

The color of the time series plot can be specified using the keyword argument color='blue'.
To add a vertical line use the .axvline() method.
Use the .set_xlabel() and .set_title() methods to specify labels.'''

#Good

ASPUS.head()
index_year = ASPUS.index.year

# Compute the mean ASPUS for each month of the year
mean_ASPUS_by_year = ASPUS.groupby(index_year).mean()


#index_month = ASPUS.index.month
#
## Compute the mean ASPUS for each month of the year
#sum_ASPUS_by_month = ASPUS.groupby(index_month).sum()
#GOOD

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
from matplotlib.pyplot import text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Plot the time series in your dataframe
ax = ASPUS.plot(color='blue', fontsize=12)

# Specify the labels in your plot
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Average Sales Price of Houses Sold for the United States', fontsize=12)

# Annotate
x_line_annotation = dt.datetime(2006, 12, 1)

# Annotate

x_text_annotation = dt.datetime(2007, 1, 15)

# Add a red vertical line at the date 1955-12-01
ax.axvline(x=x_line_annotation, color='red', linestyle='--', alpha=0.7)
ax.text(x=x_text_annotation, y=296600, s='US Housing Crash 2007', alpha=0.7, fontsize=14, color='#334f8d', rotation=90)

x_line_annotation1 = dt.datetime(2008, 12, 1)

# Annotate

x_text_annotation1 = dt.datetime(2009, 1, 15)

# Add a red vertical line at the date 1955-12-01
ax.axvline(x=x_line_annotation1, color='red', linestyle='--', alpha=0.7)
ax.text(x=x_text_annotation1, y=246600, s='US Housing Recovery', alpha=0.7, fontsize=14, color='#334f8d', rotation=90)
plt.show()


## Annotate      via arroe=w
ax.annotate('Housing Bubble in US',
            xy=(x_line_annotation, 290600),
            xycoords='data',
            xytext= (150000,250000),
            textcoords='offset points',
            arrowprops=dict(headwidth=17, width=5, color='#363d46', connectionstyle="angle13,angleA=18,angleB=-90"),
            fontsize=12, rotation=87)

plt.show()


'''The number of airline passengers has risen a lot over time. Can you find any interesting patterns in this time series?'''

#############################
'''Nice! Make sure to always consider how readable your plots are before sharing them.'''

'''
Add summary statistics to your time series plot
It is possible to visualize time series plots and numerical summaries on one single graph by using the pandas API to matplotlib along with the table method:

# Plot the time series data in the DataFrame
ax = df.plot()

# Compute summary statistics of the df DataFrame
df_summary = df.describe()

# Add summary table information to the plot
ax.table(cellText=df_summary.values,
         colWidths=[0.3]*len(df.columns),
         rowLabels=df_summary.index,
         colLabels=df_summary.columns,
         loc='top')


Assign all the values in aspus_mean to the cellText argument.
Assign all the values in index of aspus_mean to the rowLabels argument.
Assign the column names of aspus_mean to the colLabels argument.

To access all the values of aspus_mean, use the .values attribute.
To access the index of aspus_mean, use the .index attribute.
To access the column names of aspus_mean, use the .columns attribute.
'''

#Good Stat in Table

# Plot the meat data
ax = ASPUS.plot(fontsize=10, linewidth=2)

#Converting Series to Data Frame

aspus_summary  = ASPUS.describe()
aspus_mean = aspus_summary.mean()
type(aspus_mean)

aspus_mean = aspus_mean.to_frame()
type(aspus_mean)

# Add x-axis labels
ax.set_xlabel('Date', fontsize=6)

# Add summary table information to the plot
ax.table(cellText=aspus_mean.values,
         colWidths = [0.35]*len(aspus_mean.columns),
         rowLabels=aspus_mean.index,
         colLabels=aspus_mean.columns,
         loc='top')

# Specify the fontsize and location of your legend
ax.legend(loc='upper left', bbox_to_anchor=(.68, 1.85), ncol=4, fontsize=15)

# Show plot
plt.show()

'''Great! Enhancing plots with data is usually a good way to communicate more information.'''

#############################

'''Plot your time series on individual plots
It can be beneficial to plot individual time series on separate graphs as this may improve clarity and provide more context around each time series in your DataFrame.

It is possible to create a "grid" of individual graphs by "faceting" each time series by setting the subplots argument to True. In addition, the arguments that can be added are:

layout: specifies the number of rows x columns to use.
sharex and sharey: specifies whether the x-axis and y-axis values should be shared between your plots.

Create a facetted plot of the meat DataFrame using a layout of 2 rows and 4 columns.
Ensure that the subgraphs do not share x-axis and y-axis values.

You can create a facetted plot by setting subplots to True.
To specify a layout of m rows and n columns, set layout to (m, n).
sharex and sharey should be set to False in order for the axes to not share the values.'''

# Create a facetted graph with 2 rows and 4 columns


ASPUS.plot(subplots=True,
          layout=(2, 4),
          sharex=False,
          sharey=False,
          colormap='Dark2',
          fontsize=2,
          legend=True,
          linewidth=0.2)

plt.show()

########################

# Import seaborn library
import seaborn as sns

# Get correlation matrix of the meat DataFrame
corr_aspus = ASPUS.corr(method='spearman')

# Customize the heatmap of the corr_meat correlation matrix
sns.heatmap(corr_aspus,
            annot=True,
            linewidths=0.4,
            annot_kws={"size": 10})

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


####################################

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
from matplotlib.pyplot import text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#####################
ASPUS_M.head()
ASPUS_M.info()
ax = ASPUS_M['ASPUS_3A_PCT_CHG'].plot(colormap='Spectral', fontsize=6, linewidth=0.8)


#ASPUS_M['DATE'] = pd.DatetimeIndex(ASPUS_M['DATE'])

ax = ASPUS_M['ASPUS_3A_PCT_CHG'].plot(color='blue', fontsize=12)

# Specify the labels in your plot
ax.set_xlabel('Date', fontsize=12)
#ax.set_title('Number of Monthly Airline Passengers', fontsize=12)


# Annotate
x_line_annotation = '1/1/2007'

# Annotate

x_text_annotation = '1/1/2010'



# Add a red vertical line at the date 1955-12-01
ax.axvline(x=x_line_annotation, color='red', linestyle='--', alpha=0.7)
ax.text(x=x_text_annotation, y=.3, s='Cut off Points', alpha=0.7, fontsize=12, color='#334f8d', rotation=90)

# Set labels and legend
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Percent Changes', fontsize=10)
ax.set_title('Ave US Housing Price Percent Change in 12, 24 & 36', fontsize=10)
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# Annotate your plots with vertical lines
ax.axvline('1/1/2007', color='blue', linestyle='--', linewidth=0.8)
ax.axvline('1/1/2010', color='blue', linestyle='--', linewidth=0.8)

# Show plot
plt.show()
#######################

# Plot the time series in your dataframe
ax = ASPUS_M['ASPUS_3A_PCT_CHG'].plot(title="Ave US Housing Price Percent Change in 36 Months ",legend=True)

# Specify the labels in your plot
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Housing Bubble', fontsize=12)


# Annotate
x_line_annotation = dt.datetime(2007, 1, 1)

# Annotate

x_text_annotation = dt.datetime(2007, 1, 15)

# Add a red vertical line at the date 1955-12-01
ax.axvline(x=x_line_annotation, color='red', linestyle='--', alpha=0.7)
ax.text(x=x_text_annotation, y=600, s='Cut off Points', alpha=0.7, fontsize=12, color='#334f8d', rotation=90)
plt.show()


## Annotate      via arroe=w
ax.annotate('Housing Buble 2007',
            xy=(x_line_annotation, 600),
            xycoords='data',
            xytext=(-15,-50),
            textcoords='offset points',
            arrowprops=dict(headwidth=7, width=2, color='#363d46', connectionstyle="angle3,angleA=0,angleB=-90"),
            fontsize=12, rotation=90)

plt.show()
#####################

RPCE_Q = pd.read_csv('C:/scripts/capstone2/RPCE_Q_A.csv', index_col=0)

RPCE_Q.index = pd.to_datetime(RPCE_Q.index)
RPCE_M = RPCE_Q.reindex(pd.date_range(start='19620101', end='20181201', freq='1MS'))
RPCE_M.head(20)

RPCE_M = RPCE_M.interpolate(method='linear')
RPCE_M.head(20)

RPCE_M.tail(20)

RPCE_M.to_csv('C:/scripts/capstone2/RPCE_M2.csv')

RPCE_M = pd.read_csv('C:/scripts/capstone2/RPCE_M.csv', index_col=0)

#df.pct_change(periods = 6)
RPCE_Q = RPCE_M.pct_change(periods = 3)
RPCE_M.plot()
RPCE_M.info()

RPCE_M['RPCE_2A_PCT_CHG'] = RPCE_M['RPCE_2A_PCT_CHG'].apply(pd.to_numeric)


RPCE_M_2A = RPCE_M['RPCE_2A_PCT_CHG']

data_lc['lo_code'] = data_lc['Loan Officer Name'].astype('category')

RPCE_M_2A.tail()

RPCE_M_2A.plot()


#####################Deel learn  CNN  ##################


from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

data = housing_df.as_matrix()
print(data)
data


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

#PYTHONPATH=${HOME}/anaconda/envs/python3/lib/python3.5/site-packages





#https://github.com/SnehJain/Deep-Neural-Networks-For-Stock-Price-Prediction


'''Cross-validation with shuffling
As you'll recall, cross-validation is the process of splitting your data into training and test sets multiple times. Each time you do this, you choose a different training and test set. In this exercise, you'll perform a traditional ShuffleSplit cross-validation on the company value data from earlier. Later we'll cover what changes need to be made for time series data. The data we'll use is the same historical price data for several large companies.

An instance of the Linear regression object (model) is available in your workspace along with the function r2_score() for scoring. Also, the data is stored in arrays X and y. We've also provided a helper function (visualize_predictions()) to help visualize the results.

Initialize a ShuffleSplit cross-validation object with 10 splits.
Iterate through CV splits using this object. On each iteration:
Fit a model using the training indices.
Generate predictions using the test indices, score the model (R^2) using the predictions, and collect the results.

Use the ShuffleSplit() function with n_splits argument.
Use the .split() method of the cross-validation object to yield training indices (for fitting the model) and test indices (for scoring the model).
The r2_score function should take the actual and the predicted values as inputs, and returns the score.'''
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

def visualize_predictions(results):
    fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

    # Loop through our model results to visualize them
    for ii, (prediction, score, indices) in enumerate(results):
        # Plot the predictions of the model in the order they were generated
        offset = len(prediction) * ii
        axs[0].scatter(np.arange(len(prediction)) + offset, prediction, label='Iteration {}'.format(ii))

        # Plot the predictions of the model according to how time was ordered
        axs[1].scatter(indices, prediction)
    axs[0].legend(loc="best")
    axs[0].set(xlabel="Test prediction number", title="Predictions ordered by test prediction number")
    axs[1].set(xlabel="Time", title="Predictions ordered by time")
    plt.show()



# Split our data into training and test sets
print('r2_score for h_m_s_df (Transformed RAW Hosing Data) :')
X = np.array(h_m_s_df.drop(['ASPUS_M'],1))
y = np.array(h_m_s_df['ASPUS_M'])

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.297, random_state=42)#


from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    X_train, X_test = X[tr], X[tt]
    y_train, y_test = y[tr], y[tt]
    model = Ridge()
    model.fit(X_train, y_train)

    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X_test)
    score = r2_score(y_test, prediction)
    results.append((prediction, score, tt))
    score = r2_score(y_test, prediction)
    print(score)

# Custom function to quickly visualize predictions
visualize_predictions(results)


############################
housing_df.info()

# Split our data into training and test sets
print('r2_score for h_m_s_df (Three years price differences) :')
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.297, random_state=42)#


from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

model = Ridge()
# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    X_train, X_test = X[tr], X[tt]
    y_train, y_test = y[tr], y[tt]
    
    model.fit(X_train, y_train)

    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X_test)
    score = r2_score(y_test, prediction)
    results.append((prediction, score, tt))
    score = r2_score(y_test, prediction)
    print(score)

# Custom function to quickly visualize predictions
visualize_predictions(results)



############################

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion 
#from sklearn.model_selection import train_test_split


def pretty_print_linear(coefs, names = None, sort = False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split_data(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    return X_train, X_test, Y_train, Y_test


def root_mean_square_error(y_pred,y_test):
    rmse_train = np.sqrt(np.dot(abs(y_pred-y_test),abs(y_pred-y_test))/len(y_test))
    return rmse_train



def plot_real_vs_predicted(y_pred,y_test):
    plt.plot(y_pred,y_test,'ro')
    plt.plot([0,50],[0,50], 'g-')
    plt.xlabel('predicted')
    plt.ylabel('real')
    plt.title('real_vs_predicted           .@achowdhu')
    plt.show()
    return plt


#X,Y,names = load_data()

np.set_printoptions(precision=2, linewidth=100, suppress=True, edgeitems=2)

X[0:5]
X[10:15]
y[0:5]



X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.297, random_state=42)#

X = scale_data(X)

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=6, random_state=1)



model = LinearRegression()
# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    X_train, X_test = X[tr], X[tt]
    y_train, y_test = y[tr], y[tt]
    
    model.fit(X_train, y_train)

    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X_test)
    score = r2_score(y_test, prediction)
    results.append((prediction, score, tt))
    score = r2_score(y_test, prediction)
    print(score)
    # Print the root mean square error
    
    print("Linear model: ", pretty_print_linear(model.coef_, sort = True))
    plot_real_vs_predicted(y_test, prediction)

'''Good job! This time, the predictions generated within each CV loop look 'smoother' than they were before - they look more like a real time series because you didn't shuffle the data. This is a good sanity check to make sure your CV splits are correct.'''

lasso = Lasso(alpha=.3)

# Train the model using the training sets
lasso.fit(X_train, y_train)

print("Lasso model: ", pretty_print_linear(lasso.coef_, sort = True))

# Predict the values using the model
y_lasso_predict = lasso.predict(X_test)

# Print the root mean square error
print("Lasso model - Root Mean Square Error: ", root_mean_square_error(y_lasso_predict,y_test))
plot_real_vs_predicted(y_test,y_lasso_predict)


'''   Now let's try to do regression via Elastic Net.  '''

elnet = ElasticNet(fit_intercept=True, alpha=.3)

# Train the model using the training sets
elnet.fit(X_train, y_train)

print("Elastic Net model: ", pretty_print_linear(elnet.coef_, sort = True))

# Predict the values using the model
y_elnet_predict = elnet.predict(X_test)

# Print the root mean square error
print("Elastic Net - Root Mean Square Error: ", root_mean_square_error(y_elnet_predict,y_test))
plot_real_vs_predicted(y_test,y_elnet_predict)

'''  Now let's try to do regression via Stochastic Gradient Descent.  '''
sgdreg = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)

# Train the model using the training sets
sgdreg.fit(X_train, y_train)

print("Stochastic Gradient Descent model: ", pretty_print_linear(sgdreg.coef_, sort = True))

# Predict the values using the model
y_sgdreg_predict = sgdreg.predict(X_test)

# Print the root mean square error
print("Stochastic Gradient Descent - Root Mean Square Error: ", root_mean_square_error(y_sgdreg_predict,y_test))
plot_real_vs_predicted(y_test,y_sgdreg_predict)

#######################

'''Time-based cross-validation
Finally, let's visualize the behavior of the time series cross-validation iterator in scikit-learn. Use this object to iterate through your data one last time, visualizing the training data used to fit the model on each iteration.

An instance of the Linear regression model object is available in your workpsace. Also, the arrays X and y (training data) are available too.

Import TimeSeriesSplit from sklearn.model_selection.
Instantiate a time series cross-validation iterator with 10 splits.
Iterate through CV splits. On each iteration, visualize the values of the input data that would be used to train the model for that iteration.

Import TimeSeriesSplit from sklearn.model_selection
Initialize it with TimeSeriesSplit(10)
You need to loop over cv.split(X, y).'''

# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    X_train, X_test = X[tr], X[tt]
    y_train, y_test = y[tr], y[tt]
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y_train)

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()

'''Note that the size of the training set grew each time when you used the time series cross-validation object. This way, the time points you predict are always after the timepoints we train on.'''


#############

'''Bootstrapping a confidence interval
A useful tool for assessing the variability of some data is the bootstrap. Our own bootstrapping function that can be used to return a bootstrapped confidence interval.

This function takes three parameters: a 2-D array of numbers (data), a list of percentiles to calculate (percentiles), and the number of boostrap iterations to use (n_boots). It uses the resample function to generate a bootstrap sample, and then repeats this many times to calculate the confidence interval.

The function should loop over the number of bootstraps (given by the parameter n_boots) and:
Take a random sample of the data, with replacement, and calculate the mean of this random sample
Compute the percentiles of bootstrap_means and return it

To randomly sample data with replacement, use the resample() function from sklearn.utils.
To calculate the percentiles, use np.percentile().'''

from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        X_train, X_test = X[tr], X[tt]
        y_train, y_test = y[tr], y[tt]
        # Generate random indices for data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)

    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    print('\nbootstrap_means, percentiles :', percentiles)
    return percentiles

'''You can use this function to assess the variability of your model coefficients.'''

'''Calculating variability in model coefficients
In this lesson, you'll re-run the cross-validation routine used before, but this time paying attention to the model's stability over time. You'll investigate the coefficients of the model, as well as the uncertainty in its predictions.

Begin by assessing the stability (or uncertainty) of a model's coefficients across multiple CV splits. Remember, the coefficients are a reflection of the pattern that your model has found in the data.

An instance of the Linear regression object (model) is available in your workpsace. Also, the arrays X and y (the data) are available too.

Initialize a TimeSeriesSplit cross-validation object
Create an array of all zeros to collect the coefficients.
Iterate through splits of the cross-validation object. On each iteration:
Fit the model on training data
Collect the model's coefficients for analysis later

Use np.zeros() to create the empty array.
You can extract model coefficients using model.coef_.'''

# Iterate through CV splits
n_splits = 50
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    X_train, X_test = X[tr], X[tt]
    y_train, y_test = y[tr], y[tt]
    # Fit the model on training data and collect the coefficients
    model.fit(X_train, y_train)
    coefficients[ii] = model.coef_
    print('\ncoefficients[ii]:', coefficients[ii])



'''Finally, calculate the 95% confidence interval for each coefficient in coefficients using the bootstrap_interval() function we have defined earlier. You can run bootstrap_interval? if you want a refresher on the parameters that this function takes.

Call bootstrap_interval() on coefficients'''
#'ASPUS_3A_PCT_CHG', 
feature_names = (['H_RATIO_3A_PCT_CHG', 'HSN1F_3A_PCT_CHG', 'PERMIT_3A_PCT_CHG', 'STOCK_MKT_3A_PCT_CHG', 'BAA_YEILD_10Y_2A_PCT_CHG',
'US10Y_3A_PCT_CHG', 'RPCE_A_PCT_CHG', 'UEMP_3A_PCT_CHG', 'RGDP_M_PCT_CHG'])

# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

##########   Deep Learning

import pandas as pd
pd.set_option('display.max_rows',5000)
pd.set_option('display.max_columns',11)
pd.set_option('display.max_colwidth',30)
pd.set_option('display.width',None)

# What version of Python do you have?
import sys

import keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

h_m_df.info()

from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

data = housing_df.as_matrix()
print(data)
data


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

'''
The Rectified Linear Activation Function

As Dan explained to you in the video, an "activation function" is a function applied at each node. It converts the node's input into some output.

The rectified linear activation function (called ReLU) has been shown to lead to very high-performance networks. This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.

Here are some examples:
relu(3) = 3
relu(-3) = 0


Fill in the definition of the relu() function:
Use the max() function to calculate the value for the output of relu().
Apply the relu() function to node_0_input to calculate node_0_output.
Apply the relu() function to node_1_input to calculate node_1_output.
'''
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)

'''Great work! You predicted 52 transactions. Without this activation function, you would have predicted a negative number! The real power of activation functions will come soon when you start tuning model weights.'''

'''
Applying the network to many observations/rows of data

You'll now define a function called predict_with_network() which will generate predictions for multiple data observations, which are pre-loaded as input_data. As before, weights are also pre-loaded. In addition, the relu() function you defined in the previous exercise has been pre-loaded.


Define a function called predict_with_network() that accepts two arguments - input_data_row and weights - and returns a prediction from the network as the output.
Calculate the input and output values for each node, storing them as: node_0_input, node_0_output, node_1_input, and node_1_output.
To calculate the input value of a node, multiply the relevant arrays together and compute their sum.
To calculate the output value of a node, apply the relu() function to the input value of the node.
Use a for loop to iterate over input_data:
Use your predict_with_network() to generate predictions for each row of the input_data - input_data_row. Append each prediction to results.
'''

input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]

weights = {'node_0': array([2, 4]), 'node_1': array([ 4, -5]), 'output': array([2, 7])}
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])

    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)

    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)

'''
Making multiple updates to weights

You are now going to make multiple updates so you can dramatically improve your model weights, and see how the predictions improve with each update.

To keep your code clean, there is a pre-loaded get_slope() function that takes input_data, target, and weights as arguments. There is also a get_mse() function that takes the same arguments. The input_data, target, and weights have been pre-loaded.

This network does not have any hidden layers, and it goes directly from the input (with 3 nodes) to an output node. Note that weights is a single array.

We have also pre-loaded matplotlib.pyplot, and the error history will be plotted after you have done your gradient descent steps.

Using a for loop to iteratively update weights:
Calculate the slope using the get_slope() function.
Update the weights using a learning rate of 0.01.
Calculate the mean squared error (mse) with the updated weights using the get_mse() function.
Append mse to mse_hist.
Hit 'Submit Answer' to visualize mse_hist. What trend do you notice?

import inspect
lines = inspect.getsource(get_slope)
print(lines)

import inspect
lines = inspect.getsource(get_error)
print(lines)
'''

import numpy as np
import matplotlib.pyplot as plt

def get_error(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    return(error)

def get_mse(input_data, target, weights):
    errors = get_error(input_data, target, weights)
    mse = np.mean(errors**2)
    return(mse)

def get_slope(input_data, target, weights):
    error = get_error(input_data, target, weights)
    slope = 2 * input_data * error
    return(slope)

weights = array([0, 2, 1])
input_data = array([1, 2, 3])

target = 0


n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)

    # Update the weights: weights
    weights = weights - 0.01 * slope

    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)

    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
'''As you can see, the mean squared error decreases as the number of iterations go up.'''

#######  Model Building
'''
Specifying a model

Now you'll get to work with your first model in Keras, and will immediately be able to run more complex neural network models on larger datasets compared to the first two chapters.

To start, you'll take the skeleton of a neural network and add a hidden layer and an output layer. You'll then fit that model and see Keras do the optimization so your model continually gets better.

As a start, you'll predict workers wages based on characteristics like their industry, education and level of experience. You can find the dataset in a pandas dataframe called df. For convenience, everything in df except for the target has been converted to a NumPy matrix called predictors. The target, wage_per_hour, is available as a NumPy matrix called target.

we've imported the Sequential model constructor, the Dense layer constructor, and pandas.

Store the number of columns in the predictors data to n_cols. 
Start by creating a Sequential model called model.
Use the .add() method on model to add a Dense layer.
Add 50 units, specify activation='relu', and the input_shape parameter to be the tuple (n_cols,) which means it has n_cols items in each row of data, and any number of rows of data are acceptable as inputs.
Add another Dense layer. This should have 32 units and a 'relu' activation.
Finally, add an output layer, which is a Dense layer with a single node. Don't use any activation function here.
'''
# Import necessary modules


import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing_df.info()
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

# Split our data into training and test sets
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.295, random_state=42)

df = pd.read_csv('c:/scripts/deep_learning/hourly_wages.csv')
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

'''
predictors = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
target = np.array(housing_df['ASPUS_3A_PCT_CHG'])

print(predictors.shape)
print(target.shape)

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))
'''
Compiling the model

We're now going to compile the model you specified earlier. To compile the model, you need to specify the optimizer and loss function to use. The Adam optimizer is an excellent choice. 
Here we'll use the Adam optimizer and the mean squared error loss function. Go for it!

Compile the model using model.compile(). Your optimizer should be 'adam' and the loss should be 'mean_squared_error'.
'''
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)


'''Fantastic work - all that's left now is to fit the model!'''

'''
Fitting the model

You're at the most fun part. You'll now fit the model. Recall that the data to be used as predictive features is loaded in a NumPy matrix called predictors and the data to be predicted is stored in a NumPy matrix called target. Your model is pre-written and it has been compiled with the code from the previous exercise.

Fit the model. Remember that the first argument is the predictive features (predictors), and the data to be predicted (target) is the second argument.
'''
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)

'''You now know how to specify, compile, and fit a deep learning model using keras!'''





'''
Last steps in classification models

You'll now create a classification model using the titanic dataset, which has been pre-loaded into a DataFrame called df. You'll take information about the passengers and predict which ones survived.

The predictive variables are stored in a NumPy array predictors. The target to predict is in df.survived, though you'll have to manipulate it for keras. The number of predictive features is stored in n_cols.

Here, you'll use the 'sgd' optimizer, which stands for Stochastic Gradient Descent. You'll learn more about this in the next chapter!

Convert df.survived to a categorical variable using the to_categorical() function.
Specify a Sequential model called model.
Add a Dense layer with 32 nodes. Use 'relu' as the activation and (n_cols,) as the input_shape.
Add the Dense output layer. Because there are two outcomes, it should have 2 units, and because it is a classification model, the activation should be 'softmax'.
Compile the model, using 'sgd' as the optimizer, 'categorical_crossentropy' as the loss function, and metrics=['accuracy'] to see the accuracy (what fraction of predictions were correct) at the end of each epoch.
Fit the model using the predictors and the target.
'''
#Error

# Import necessary modules
import keras
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
print( keras.__version__ )

#from keras.activations import softmax
predictors = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
target = np.array(housing_df['ASPUS_3A_PCT_CHG'])

print(predictors.shape)
print(target.shape)

# Set up the model
model = Sequential()
#n_cols=10
# Add the first layer
#model.add(Dense(32, activation='relu', input_shape=((10),)))
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
#model.add( Dense(6, input_shape=(6,), activation = 'softmax' ) )
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['mse'])

# Fit the model
model.fit(predictors, target, epochs=20)

#history = model.fit(predictors, target, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
#train_mse = model.evaluate(predictors, target, verbose=0)
##test_mse = model.evaluate(testX, testy, verbose=0)
##print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
#print('Train: %.3f, Test: %.3f'% (train_mse))
## plot loss during training
#pyplot.title('Loss / Mean Squared Error')
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

'''This simple model is generating an accuracy of 75!'''

'''
Making predictions

The trained network from your previous coding exercise is now stored as model. New data to make predictions is stored in a NumPy array as pred_data. Use model to make predictions on your new data.

In this exercise, your predictions will be probabilities, which is the most common way for data scientists to communicate their predictions to colleagues.

Create your predictions using the model's .predict() method on pred_data.
Use NumPy indexing to find the column corresponding to predicted probabilities of survival being True. This is the second column (index 1) of predictions. Store the result in predicted_prob_true and print it.'''

new_input1 = np.array([[ 8.60000000e+00,  2.21000000e+02,  1.32600000e+03,  -7.88567149e+00,  2.30000000e+00,  3.83000000e+00,   3.80000000e+00,  4.62318061e+00,  3.60000000e+00],
       [ 2.50000000e+00,  6.85000000e+02,  19.30000000e+02,   13.10174247e+00,  7.12000000e+00,  0.72000000e+00,   4.43000000e+00,  2.79994189e+00,  9.00000000e-01],
       [12.30000000e+00,  12.07400000e+03,  11.86700000e+03,  -9.36840249e-01,  0.67000000e+00,  1.11000000e+00,   1.27000000e+00,  14.69406217e+00,  2.00000000e-01],
       [ 0.30000000e+00,  9.00000000e+02,  1.65100000e+03,   12.46476236e+00,  2.11000000e+00,  1.26000000e+00,   4.70000000e+00,  4.08157421e+00,  5.50000000e+00],
       [ 4.50000000e+00,  8.12000000e+02,  1.46100000e+03,   1.61322058e+00,  1.92000000e+00,  5.77000000e+00,   4.33000000e+00,  6.91357601e+00,  4.46666667e+00],
       [ 16.80000000e+00,  6.66000000e+02,  1.51400000e+03,  -1.06550895e+01,  1.89000000e+00,  9.42000000e+00,   2.13000000e+00,  5.76349116e+00,  5.83333333e+00],
       [ 6.20000000e+00,  2.15000000e+02,  9.76000000e+02,  2.83633570e+00,  2.33000000e+00,  2.34700000e+00,   1.27000000e+00,  3.40560669e+00,  4.30000000e+00],
       [ 18.90000000e+00,  4.77000000e+02,  7.09000000e+02,   1.34337920e+01,  2.75000000e+00,  7.73000000e+00,   5.60000000e+00,  8.90922651e+00,  3.33333333e-01],
       [5.20000000e+00,  5.11000000e+02,  1.38900000e+03,  -7.64093873e-01,  1.20000000e+00,  6.03000000e+00,   3.60000000e+00,  3.49405492e+00,  4.80000000e+00],
       [ 6.70000000e+00,  1.91000000e+02,  1.22300000e+03,  -6.97363277e-01,  1.05000000e+00,   4.98000000e+00,   3.93000000e+00,  4.12848311e+00,  2.53333333e+00]])
    
X[::-10]

input_data1 = np.array([[   6.6 ,  621.  , 1326.  ,   -5.89,    2.3 ,    2.83,    2.8 ,    3.62,    2.6 ],
       [   5.4 ,  663.  , 1323.  ,    1.44,    1.65,    2.86,    1.6 ,    4.21,    2.87],
       [   5.4 ,  593.  , 1255.  ,    1.81,    2.27,    2.3 ,    2.9 ,    4.28,    3.  ],
       [   5.2 ,  560.  , 1194.  ,    4.71,    2.89,    1.64,    2.93,    4.97,    2.03],
       [   5.1 ,  515.  , 1159.  ,   -4.24,    3.02,    2.17,    2.7 ,    5.15,    0.8 ],
       [   5.3 ,  472.  , 1099.  ,   -1.66,    2.39,    2.3 ,    4.7 ,    5.54,    1.9 ],
       [   5.2 ,  433.  , 1009.  ,    3.44,    2.48,    2.9 ,    2.23,    6.98,    0.4 ],
       [   4.1 ,  447.  ,  979.  ,    6.43,    2.87,    1.98,    1.5 ,    7.96,    2.57],
       [   4.9 ,  354.  ,  732.  ,   -2.73,    3.14,    2.05,    0.6 ,    8.11,    1.7 ],
       [   6.6 ,  301.  ,  636.  ,   -5.66,    2.75,    3.  ,    1.37,    9.12,    0.9 ],
       [   8.8 ,  282.  ,  580.  ,    1.72,    2.96,    2.7 ,    2.77,    9.5 ,    2.67],
       [   7.4 ,  396.  ,  583.  ,    9.37,    2.9 ,    3.39,   -0.6 ,    9.64,    4.5 ],
       [  11.2 ,  377.  ,  554.  ,  -17.51,    6.01,    2.42,   -1.83,    8.17,   -5.73],
       [   9.7 ,  593.  , 1014.  ,   -5.33,    3.08,    3.74,   -0.53,    5.32,   -0.83],
       [   7.4 ,  887.  , 1470.  ,    6.04,    1.7 ,    4.69,    0.7 ,    4.48,    2.3 ],
       [   6.3 , 1074.  , 1867.  ,    0.94,    1.67,    5.11,    2.27,    4.69,    0.7 ],
       [   4.5 , 1255.  , 2219.  ,    3.67,    1.7 ,    4.26,    3.  ,    4.93,    3.27],
       [   3.9 , 1305.  , 2097.  ,    6.89,    2.11,    4.1 ,    4.4 ,    5.2 ,    4.1 ],
       [   4.  , 1129.  , 1987.  ,    8.57,    2.33,    4.27,    3.3 ,    5.99,    3.03],
       [   4.5 ,  936.  , 1854.  ,    0.67,    3.16,    3.9 ,    2.8 ,    6.31,    2.63],
       [   4.3 ,  936.  , 1669.  ,   -2.54,    2.82,    5.21,    2.1 ,    5.82,    2.4 ],
       [   4.2 ,  882.  , 1626.  ,   -4.33,    2.69,    5.28,    1.23,    4.69,   -0.33],
       [   4.4 ,  848.  , 1552.  ,    0.93,    2.43,    5.83,    3.83,    3.93,    1.17],
       [   4.2 ,  872.  , 1649.  ,   -0.33,    2.27,    6.11,    6.  ,    3.84,    7.  ],
       [   3.8 ,  949.  , 1742.  ,    5.89,    2.58,    4.65,    4.57,    4.51,    4.73],
       [   3.9 ,  866.  , 1647.  ,    7.12,    1.68,    5.57,    5.13,    4.87,    4.  ],
       [   4.7 ,  744.  , 1421.  ,    4.24,    1.45,    6.89,    1.8 ,    4.96,    6.8 ],
       [   6.  ,  721.  , 1429.  ,    1.27,    1.49,    6.91,    3.07,    5.34,    4.67],
       [   6.1 ,  701.  , 1386.  ,    6.74,    1.7 ,    6.49,    3.4 ,    5.53,    3.23],
       [   5.6 ,  715.  , 1397.  ,   -0.87,    1.46,    7.74,    4.4 ,    5.31,    4.7 ],
       [   4.5 ,  812.  , 1461.  ,    1.61,    1.92,    5.77,    4.33,    6.91,    4.47],
       [   5.3 ,  604.  , 1148.  ,    3.35,    2.13,    6.26,    2.23,    7.61,    1.23],
       [   6.1 ,  546.  , 1054.  ,   -0.71,    1.73,    7.48,    2.7 ,    7.59,    4.4 ],
       [   7.  ,  516.  ,  964.  ,    3.77,    1.68,    8.28,    2.47,    6.8 ,    2.4 ],
       [   8.2 ,  528.  , 1069.  ,   -4.5 ,    1.66,    8.75,    0.07,    5.7 ,   -1.  ],
       [   6.9 ,  645.  , 1365.  ,    0.44,    1.8 ,    8.01,    1.8 ,    5.16,    0.8 ],
       [   6.8 ,  658.  , 1501.  ,    4.54,    1.54,    9.11,    2.8 ,    5.52,    4.53],
       [   6.6 ,  663.  , 1438.  ,    1.8 ,    2.41,    8.21,    5.87,    6.  ,    3.2 ],
       [   6.  ,  735.  , 1601.  ,    3.9 ,    2.02,    8.02,    5.6 ,    6.31,    4.4 ],
       [   5.7 ,  728.  , 1790.  ,    3.19,    2.54,    7.8 ,    6.33,    7.02,    3.2 ],
       [   5.8 ,  726.  , 1808.  ,    2.65,    2.17,   10.33,    5.5 ,    7.05,    5.13],
       [   6.  ,  687.  , 1490.  ,    3.21,    1.78,   12.16,    5.3 ,    7.06,    3.3 ],
       [   4.7 ,  773.  , 1627.  ,   -2.26,    1.92,   11.83,    4.43,    8.49,    8.27],
       [   5.6 ,  562.  , 1471.  ,    8.87,    3.23,   10.72,    5.47,   10.97,    6.73],
       [   9.4 ,  339.  ,  888.  ,   -0.41,    2.91,   13.87,    1.2 ,    9.49,    1.8 ],
       [   9.2 ,  415.  ,  976.  ,   -2.84,    2.33,   13.47,    1.27,    7.41,    2.3 ],
       [   6.  ,  659.  , 1367.  ,   12.55,    2.05,   11.1 ,    4.83,    7.5 ,    2.23],
       [   7.4 ,  670.  , 1481.  ,   -0.78,    1.1 ,   10.3 ,    1.  ,    5.73,    1.  ],
       [   6.1 ,  805.  , 1818.  ,   -0.  ,    0.93,    9.01,    2.5 ,    6.14,    2.3 ],
       [   6.3 ,  791.  , 1736.  ,    0.05,    1.17,    8.03,    4.47,    6.59,    6.33],
       [   5.5 ,  799.  , 1660.  ,   -2.23,    1.7 ,    7.37,    2.2 ,    7.01,    8.  ],
       [   6.9 ,  591.  , 1188.  ,    2.65,    2.03,    7.86,    4.1 ,    7.61,    2.47],
       [   6.8 ,  566.  ,  994.  ,   -0.53,    2.19,    8.4 ,    5.33,    8.24,    6.5 ],
       [   9.8 ,  448.  ,  824.  ,   -7.41,    2.58,    7.9 ,   -5.7 ,    6.24,   -1.5 ],
       [  10.1 ,  519.  , 1288.  ,   -5.12,    1.74,    6.74,   -2.73,    5.26,   -1.  ],
       [   6.8 ,  737.  , 2226.  ,   -3.2 ,    1.33,    6.64,    4.93,    5.31,    8.33],
       [   5.8 ,  684.  , 2139.  ,    2.71,    2.05,    6.19,    7.8 ,    5.68,    9.4 ],
       [   4.7 ,  646.  , 1903.  ,   -0.  ,    2.23,    6.52,    3.37,    6.02,    2.93],
       [   5.1 ,  515.  , 1394.  ,    2.23,    1.91,    7.53,    1.97,    5.34,    1.07],
       [   7.  ,  401.  , 1216.  ,    0.72,    1.12,    7.1 ,    3.2 ,    3.39,   -1.9 ],
       [   5.2 ,  511.  , 1389.  ,   -0.76,    1.2 ,    6.03,    3.6 ,    3.49,    4.8 ],
       [   4.1 ,  543.  , 1342.  ,    0.69,    1.24,    5.56,    8.67,    3.89,    7.9 ],
       [   4.9 ,  479.  , 1035.  ,    6.04,    1.24,    4.59,    5.6 ,    3.81,    0.2 ],
       [   6.  ,  434.  ,  956.  ,   -6.44,    0.77,    4.81,    3.47,    3.78,    2.73],
       [   4.6 ,  615.  , 1249.  ,    1.41,    0.63,    4.25,    8.57,    4.07,    9.3 ],
       [   5.3 ,  583.  , 1230.  ,    2.63,    0.62,    4.19,    1.1 ,    4.58,    1.2 ],
       [   6.1 ,  514.  , 1423.  ,    5.44,    0.72,    4.13,    6.53,    5.87,    6.67],
       [   6.6 ,  464.  , 1212.  ,    8.  ,    0.97,    3.92,    3.13,    6.4 ,    4.47],
       [   4.7 ,  591.  , 1235.  ,  -10.53,    1.18,    3.84,    5.  ,    5.54,    3.7 ]])




############

predictors = np.array(h_m_df.drop(['ASPUS_M'],1))
target = np.array(h_m_df['ASPUS_M'])

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = ((n_cols),)))
model.add(Dense(1, activation='softmax'))
#model.compile(optimizer='sgd',              loss='categorical_crossentropy',              metrics=['accuracy'])

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['mse'])

model.fit(predictors, target, epochs=30)

# Calculate predictions: predictions
predictions = model.predict(input_data1)

# Calculate predicted probability of survival: predicted_prob_true (2nd col)
predicted_prob_true = predictions[:,1]
###########
# print predicted_prob_true
print(predicted_prob_true)

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = ((n_cols),)))
model.add(Dense(1, activation='softmax'))
#model.compile(optimizer='sgd',              loss='categorical_crossentropy',              metrics=['accuracy'])

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['mse'])

model.fit(predictors, target, epochs=30)

# Calculate predictions: predictions
predictions = model.predict(input_data1)

# Calculate predicted probability of survival: predicted_prob_true (2nd col)
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

'''You're now ready to begin learning how to fine-tune your models.

You have finished the chapter "Building deep learning models with keras"!'''
####################
#Good
# mlp for regression with mae loss function
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
# generate regression dataset
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)
# standardize dataset
# split into train and test
n_cols=9
# define model
model = Sequential()
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', input_shape = ((n_cols),)))
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.001, momentum=0.99)
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])

# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)
# evaluate the model
_, train_mse = model.evaluate(X_train, y_train, verbose=0)
_, test_mse = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot mse during training
pyplot.subplot(212)
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['mean_squared_error'], label='train')
pyplot.plot(history.history['val_mean_squared_error'], label='test')
pyplot.legend()
pyplot.show()



#HERE

###   Housing price Prediction_Using_CNN

h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df.csv', index_col='DATE', parse_dates=True)
h_m_df.head()
#Good
# mlp for regression with mae loss function
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from sklearn import metrics
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping

# generate regression dataset
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(h_m_df)
#type(model)
h_m_df.info()



X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])


#######


X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
y = sc.fit_transform(y.reshape(len(y),1))[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)
# built Keras sequential model 



'''
2.2 Split data to training set and test set
Split data into training (60%) and test sets (40%).
'''



data = h_m_df.as_matrix()
df = []
for index in range(len(data) - 31):
    df.append(data[index: index + 31])
df = np.array(df)

splitRow = round(0.6 * df.shape[0])

## Training (60%)
train = df[:int(splitRow), :]
X_train = train[:, :-1] # all data until day 30
y_train = train[:, -1][:,-1] # day 31 close price
print('Training set:', train.shape[0], 'obs')
​
## Test (40%)
test = df[int(splitRow):, :]
X_test = test[:, :-1] # all data until day 30
y_test = test[:, -1][:,-1] # day 31 close price
print('Test set:', test.shape[0], 'obs')

h_m_df.shape
print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)



'''
3. Modeling
Now predict the close price for day 
𝑚
m
based on data observed in the past 30 days 
{𝑚−30,𝑚−29,...𝑚−1}
{m−30,m−29,...m−1}
.
3.1 Define Network
Define a Sequential Model and add:
input layer with dimension (30, 6);
two LSTM layers with 256 neurons;
one hidden layers with 32 neurons with 'Relu';
one linear output layer.'''




from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.optimizers import Adam 

# Split into train/test
model = Sequential()
model.add(Dense(100, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(50, activation='relu')) # Hidden 2
model.add(Dense(1,activation='softmax')) # Output

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(X,y,verbose=1,epochs=100)

model.summary()

print(model.summary())

######

#####

'''
3.2 Compile Network¶
Before training, configure the learning process by specifying:
Optimizer to be 'adam';
Loss function to be 'mse';
Evaluation metric to be 'accuracy'.
'''
decay = .001
import tensorflow as tf
import keras
adam = keras.optimizers.Adam(decay=decay)
model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

'''
3.3 Train Network¶
Fit the model to training data to learn the parameters.
'''

model.fit(X_train, y_train,
    batch_size=512,
    epochs=100,
    validation_split=0.2,
    verbose=2)

'''
3.4 Evaluate Network
Evaluate model on the test set
'''
mse = model.evaluate(X_test, y_test)
print("mean square error = ", mse)

#X_test = X_test.reshape(-1,1)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error", 'accuracy', r_sq])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=40, verbose=1, mode='auto')
result1= model.fit(X,y,verbose=2,epochs=120)
#, batch_size=5, validation_data=(X_test, y_test)
# fit model
result = model.fit(X_train,y_train,verbose=2,epochs=120, batch_size=5, validation_data=(X_test, y_test), callbacks=[earlystopping])
# get predictions
y_pred = model.predict(X_test)
#####

from keras import backend as K

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


def r_sq(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------
# plot training curve for R^2 (beware of scale, starts very low negative)
#plt.plot(result1.history['val_r_square'])
plt.plot(result.history['r_sq'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result1.history['rmse'])
#plt.plot(result1.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


##############

#LSTM

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from keras import backend as K



# generate regression dataset
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score 

import pandas as pd
import numpy as np
h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df2.csv')
h_m_df.head()
h_m_df.info()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

n_sample = h_m_df.shape[0]

n_train=int(0.8*n_sample)+1
n_forecast=n_sample-n_train
#ts_df
dataset_train = h_m_df.iloc[:n_train]['ASPUS_M']
dataset_test = h_m_df.iloc[n_train:]['ASPUS_M']
print(dataset_train.shape)
print(dataset_test.shape)
print("Training Series Tail:", "\n", dataset_train.tail(), "\n")
print("Testing Series Head:", "\n", dataset_test.head())

'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# Recurrent Neural Network
type(X_train)'''
type(dataset_train)

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
#dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#training_set = ('X_train', 'y_train')
training_set = dataset_train.values.reshape(-1, 1)
test_set = dataset_test.values.reshape(-1, 1)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(6, 548):
    X_train.append(training_set_scaled[i-6:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 75, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real Housing price of 2017

real_housing_price = test_set
#dataset_train['ASPUS_M']
# Getting the predicted Housing price of 2017
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 6:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(6, 143):
    X_test.append(inputs[i-6:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_housing_price = regressor.predict(X_test)
predicted_housing_price = sc.inverse_transform(predicted_housing_price)

# Visualising the results
plt.plot(real_housing_price, color = 'red', label = 'Real US Ave Housing Price')
plt.plot(predicted_housing_price, color = 'blue', label = 'Predicted US Ave Housing Price')
plt.title('US Housing Market Ave Price Prediction - Usning Recurrent Neural Network (LSTM)')
plt.xlabel('Time in Months')
plt.ylabel('US Housing Price')
plt.legend()
plt.show()

'''  END LSTM '''
##########
#REPEAT

#LSTM

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
# generate regression dataset
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score 

import pandas as pd
import numpy as np
h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df.csv')
h_m_df.head()
h_m_df.info()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

n_sample = h_m_df.shape[0]

n_train=int(0.8*n_sample)+1
n_forecast=n_sample-n_train
#ts_df
dataset_train = h_m_df.iloc[:n_train]['ASPUS_M']
dataset_test = h_m_df.iloc[n_train:]['ASPUS_M']
print(dataset_train.shape)
print(dataset_test.shape)
print("Training Series Tail:", "\n", dataset_train.tail(), "\n")
print("Testing Series Head:", "\n", dataset_test.head())

'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# Recurrent Neural Network
'''
type(dataset_train)

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
#dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#training_set = ('X_train', 'y_train')
training_set = dataset_train.values.reshape(-1, 1)
test_set = dataset_test.values.reshape(-1, 1)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
test_set_scaled = sc.fit_transform(test_set)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(6, 548):
    X_train.append(training_set_scaled[i-6:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
y_test = []
for i in range(6, 136):
    X_test.append(test_set_scaled[i-6:i, 0])
    y_test.append(test_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU

from keras import backend as K

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


def r_sq(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_sq])

decay = .001
import tensorflow as tf
import keras
adam = keras.optimizers.Adam(decay=decay)

# compile regression model loss should be mean_squared_error //


# Initialising the RNN
regressor = Sequential()
#regressor.add(BatchNormalization())
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 200, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.3))


# now add a ReLU layer explicitly:
regressor.add(LeakyReLU(alpha=0.05))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 150,  return_sequences = True)) #activation='sigmoid'
regressor.add(Dropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# Adding the output layer
regressor.add(Dense(units = 1, activation='sigmoid')) #, activation='sigmoid'

# Compiling the RNN
#regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_sq])

#opt = SGD(lr=0.0001, momentum=0.9, decay=0.01)

regressor.compile(optimizer='adam', loss=rmse, metrics=[rmse, r_sq])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="r_sq", patience=100, verbose=1, mode='auto')
# fit model validation_data=(X_test, y_test)
result = regressor.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[earlystopping])

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)

print(regressor.summary())

print(regressor.input_shape)
print(regressor.output_shape)


# root mean squared error (rmse) for regression (only for Keras tensors)
# Get accuracy of model on validation data. It's not AUC but it's something at least!

score = regressor.evaluate(X_test,y_test, batch_size=32)
print('Test accuracy:', score)
# Part 3 - Making the predictions and visualising the results

# Getting the real Housing price of 2017

real_housing_price = test_set
#dataset_train['ASPUS_M']
# Getting the predicted Housing price of 2017
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 6:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(6, 142):
    X_test.append(inputs[i-6:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_housing_price = regressor.predict(X_test)
predicted_housing_price = sc.inverse_transform(predicted_housing_price)

print(predicted_housing_price.shape)



# Visualising the results
plt.plot(real_housing_price, color = 'red', label = 'Real US Ave Housing Price')
plt.plot(predicted_housing_price, color = 'blue', label = 'Predicted US Ave Housing Price')
plt.title('US Housing Market Ave Price Prediction - Usning Recurrent Neural Network (LSTM)')
plt.xlabel('Time in Months')
plt.ylabel('US Housing Price')
plt.legend()
plt.show()

################

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------

# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['val_r_sq'])
plt.plot(result.history['r_sq'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#--END LSTM--------------------------------------------


##########

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics

housing_df= pd.read_csv('C:/scripts/capstone2/housing_df.csv', index_col='DATE')

X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)


# Feature Scaling
model = Sequential()
model.add(Dense(25, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(10, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(optimizer='adam', loss=mse, metrics=[mse, r_sq])
model.fit(X,y,verbose=2,epochs=100)


'''
Controling the Amount of Output
One line is produced for each training epoch. You can eliminate this output by setting the verbose setting of the fit command:
verbose=0 - No progress output (use with Juputer if you do not want output)
verbose=1 - Display progress bar, does not work well with Jupyter
verbose=2 - Summary progress output (use with Jupyter if you want to know the loss at each epoch)
Regression Prediction

Next, we will perform actual predictions. These predictions are assigned to the pred variable. These are all Housing % Change predictions from the neural network. Notice that this is a 2D array? You can always see the dimensions of what is returned by printing out pred.shape. Neural networks can return multiple values, so the result is always an array. Here the neural network only returns 1 value per prediction (there are 398 cars, so 398 predictions). However, a 2D array is needed because the neural network has the potential of returning more than one value. 
'''

pred = model.predict(X)
print("Shape: {}".format(pred.shape))
print(pred)

'''
We would like to see how good these predictions are. We know what the correct Housing % Change is for each property, so we can measure how close the neural network was.
'''

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y))
print(f"Final score (RMSE): {score}")


# Sample predictions
for i in range(60):
    i=i+60
    print(f"{(i-59)}. House 3 years growth: {y[i]}, Acctual_ASPUS_3A_PCT_CHG: {y[i]}, predicted_ASPUS_3A_PCT_CHG: {pred[i]}")


################

'''
Simple TensorFlow Classification: Housing 3 Years growth data set
This is a very simple example of how to perform the Housing 3 Years growth classification using TensorFlow. '''

import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping

X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])


# Split into train/test
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(1,activation='softmax')) # Output

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(X,y,verbose=1,epochs=100)

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_sq(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# custom function example
model.compile(optimizer="adam", loss=rmse, metrics=[r_sq, rmse])
model.fit(X,y,verbose=1,epochs=100)

######################

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#The calling convention for Keras backend functions in loss and metrics is:

# original Keras functions
model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error"])

# custom function example
model.compile(optimizer="Nadam", loss=rmse, metrics=[r_square, rmse])


"""
Created on Wed Aug 15 18:44:28 2018
Simple regression example for Keras (v2.2.2) with  housing data
@author: tobigithub
"""
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

#-----------------------------------------------------------------------------
# Start a simple Keras sequential model
#-----------------------------------------------------------------------------
#Good
# set the seeds for reproducible results with TF (wont work with GPU, only CPU)
np.random.seed(12345)
# set the TF seed
set_random_seed(12345)
# Import data, assign seed for same results, do train/test split 80/20

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from keras import backend as K

def r_sq(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# generate regression dataset
from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y = np.array(housing_df['ASPUS_3A_PCT_CHG'])

X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)

# built Keras sequential model 
model = Sequential()
# add batch normalization
model.add(BatchNormalization())
# add layer to the MLP for data (404,13) 

model.add(Dense(100, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(50, activation='relu')) # Hidden 2
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
# compile regression model loss should be mean_squared_error //
model.compile(optimizer="adam", loss=rmse, metrics=[rmse, r_square])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=40, verbose=2, mode='auto')
#result1= model.fit(X_train, y_train,verbose=2,epochs=500)
result1 = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[earlystopping])

#
# get predictions
y_pred = model.predict(X_test)

score = model.evaluate(X_test,y_test, batch_size=32)
print('Test accuracy:', score)

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------
# plot training curve for R^2 (beware of scale, starts very low negative)


plt.plot(result1.history['r_square'])
plt.plot(result1.history['val_r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['R^2_train', 'R^2_test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result1.history['rmse'])
plt.plot(result1.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['rmse_train', 'rmse_test'], loc='upper left')
plt.show()

plt.close()


plt.plot(y_test, color = 'red', label = 'Real US Housing Price 3_YEAR_PCT_CHG')
plt.plot(y_pred, color = 'blue', label = 'Predicted US Housing Price 3_YEAR_PCT_CHG')
plt.title('US Housing Market Ave Price Prediction - Usning Keras sequential model ')
plt.xlabel('Time in Months')
plt.ylabel('US Housing Price 3_YEAR_PCT_CHG')
plt.legend()
plt.show()

#############

##########

# print the linear regression and display datapoints
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(y_test.reshape(-1,1), y_pred)  
y_fit = regressor.predict(y_pred) 

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(y_test, y_pred, color='blue', label= 'data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()

#-----------------------------------------------------------------------------
# print statistical figures of merit
#-----------------------------------------------------------------------------

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))


###############

from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
#X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
#y = np.array(housing_df['ASPUS_3A_PCT_CHG'])
#
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)
model.metrics_names

trainScore = model.evaluate(X_train, y_train, verbose=0)
testScore = model.evaluate(X_test, y_test, verbose=0)

#predicting values for y_test
p = model.predict(X_test)

plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.xlabel('X Values')
plt.ylabel('Housing Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 5)
#fig.savefig('img/25/mrftestcnn.png', dpi=300)
plt.show()

p1= model.predict(X_train)

plt.plot(p1[:147],color='red', label='prediction on training samples')
X = np.array(range(147,547))
plt.plot(X,p1[147:],color = 'purple',label ='prediction on validating samples')
plt.plot(y_train,color='blue', label='y_train')
plt.xlabel('X Val')
plt.ylabel('House Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,10)
#fig.savefig('img/25/mrftraincnn.png', dpi=300)
plt.show()

#y_test
#y_pred
#ma = housing_df.rolling(window=52).mean()
y = y_test *10000
y_pred = p.reshape(137)
y_pred = y_pred * 10000

from sklearn.metrics import mean_absolute_error

print('Trainscore RMSE \tTrain Mean abs Error \tTestscore Rmse \t Test Mean abs Error')
print('%.9f \t\t %.9f \t\t %.9f \t\t %.9f' % (math.sqrt(trainScore[0]),trainScore[1],math.sqrt(testScore[0]),testScore[1]))


print('mean absolute error \t mean absolute percentage error')
print(' %.9f \t\t\t %.9f' % (mean_absolute_error(y,y_pred),(np.mean(np.abs((y - y_pred) / y)) * 100)))


Y = np.concatenate((y_train,y_test),axis = 0)
P = np.concatenate((p1,p),axis = 0)
#plotting the complete Y set with predicted values on x_train and x_test(variable p1 & p respectively given above)
#for 
plt.plot(P[:147],color='red', label='prediction on training samples')
#for validating samples
z = np.array(range(147,547))
plt.plot(z,P[147:547],color = 'black',label ='prediction on validating samples')
#for testing samples
X1 = np.array(range(547,684))
plt.plot(X1,P[547:],color = 'green',label ='prediction on testing samples(x_test)')

plt.plot(Y,color='blue', label='Y')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,12)
plt.show()

############################################################################


X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)






model = Sequential()
# add batch normalization
model.add(BatchNormalization())
# add layer to the MLP for data (404,13) 

model.add(Dense(300, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(150, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(X,y,verbose=2,epochs=100)
# compile regression model loss should be mean_squared_error //
model.compile(optimizer="Nadam", loss="mean_squared_error", metrics=["mean_squared_error", rmse, r_square])
# enable early stopping based on mean_squared_error
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=20, verbose=1, mode='auto')
result1= model.fit(X,y,verbose=2,epochs=120)
#, batch_size=5, validation_data=(X_test, y_test)
# fit model
result = model.fit(X_train,y_train,verbose=2,epochs=120, batch_size=5, validation_data=(X_test, y_test), callbacks=[earlystopping])
# get predictions
y_pred = model.predict(X_test)

########

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------
# plot training curve for R^2 (beware of scale, starts very low negative)
#plt.plot(result1.history['val_r_square'])
plt.plot(result1.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result1.history['rmse'])
#plt.plot(result1.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#############
# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['val_r_square'])
plt.plot(result.history['r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
           
# plot training curve for rmse
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
##########

# print the linear regression and display datapoints
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(y_test.reshape(-1,1), y_pred)  
y_fit = regressor.predict(y_pred) 

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(y_test, y_pred, color='blue', label= 'data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()

#-----------------------------------------------------------------------------
# print statistical figures of merit
#-----------------------------------------------------------------------------

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))


###############

from sklearn.model_selection import train_test_split # for train and test set split
from sklearn.model_selection import cross_val_score #Sklearn.model_seletion is used

# Construct data for the model
type(housing_df)
#type(model)
housing_df.info()
#X = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
#y = np.array(housing_df['ASPUS_3A_PCT_CHG'])
#
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("size of the training feature set is",X_train.shape)
print("size of the test feature set is",X_test.shape)
print("size of the training Target set is",y_train.shape)
print("size of the test Target set is",y_test.shape)
model.metrics_names

trainScore = model.evaluate(X_train, y_train, verbose=0)
testScore = model.evaluate(X_test, y_test, verbose=0)

#predicting values for y_test
p = model.predict(X_test)

plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.xlabel('X Values')
plt.ylabel('Housing Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(15, 5)
#fig.savefig('img/25/mrftestcnn.png', dpi=300)
plt.show()

p1= model.predict(X_train)

plt.plot(p1[:147],color='red', label='prediction on training samples')
X = np.array(range(147,547))
plt.plot(X,p1[147:],color = 'purple',label ='prediction on validating samples')
plt.plot(y_train,color='blue', label='y_train')
plt.xlabel('X Val')
plt.ylabel('House Value (scaled)')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,10)
#fig.savefig('img/25/mrftraincnn.png', dpi=300)
plt.show()

#y_test
#y_pred
#ma = housing_df.rolling(window=52).mean()
y = y_test 
y_pred = p.reshape(137)
y_pred = y_pred

from sklearn.metrics import mean_absolute_error

print('Trainscore RMSE \tTrain Mean abs Error \tTestscore Rmse \t Test Mean abs Error')
print('%.9f \t\t %.9f \t\t %.9f \t\t %.9f' % (math.sqrt(trainScore[0]),trainScore[1],math.sqrt(testScore[0]),testScore[1]))


print('mean absolute error \t mean absolute percentage error')
print(' %.9f \t\t\t %.9f' % (mean_absolute_error(y,y_pred),(np.mean(np.abs((y - y_pred) / y)) * 100)))


Y = np.concatenate((y_train,y_test),axis = 0)
P = np.concatenate((p1,p),axis = 0)
#plotting the complete Y set with predicted values on x_train and x_test(variable p1 & p respectively given above)
#for 
plt.plot(P[:147],color='red', label='prediction on training samples')
#for validating samples
z = np.array(range(147,547))
plt.plot(z,P[147:547],color = 'black',label ='prediction on validating samples')
#for testing samples
X1 = np.array(range(547,684))
plt.plot(X1,P[547:],color = 'green',label ='prediction on testing samples(x_test)')

plt.plot(Y,color='blue', label='Y')
plt.legend(loc='upper left')
fig = plt.gcf()
fig.set_size_inches(20,12)
plt.show()

##################   FORECASTING  ARIMA ##########

'''
Applying statistical modeling and machine learning to perform time-series forecasting.


What techniques may help answer these questions?
Statistical models

Ignore the time-series aspect completely and model using traditional statistical modeling toolbox. 
Examples. Regression-based models. 
Univariate statistical time-series modeling.
Examples. Averaging and smoothing models, ARIMA models.
Slight modifications to univariate statistical time-series modeling.
Examples. External regressors, multi-variate models.
Additive or component models.
Examples. Facebook Prophet package.
Structural time series modeling.
Examples. Bayesian structural time series modeling, hierarchical time series modeling.

Machine learning models

Ignore the time-series aspect completely and model using traditional machine learning modeling toolbox. 
Examples. Support Vector Machines (SVMs), Random Forest Regression, Gradient-Boosted Decision Trees (GBDTs).
Hidden markov models (HMMs).
Other sequence-based models.
Gaussian processes (GPs).
Recurrent neural networks (RNNs).

Additional data considerations before choosing a model

Whether or not to incorporate external data
Whether or not to keep as univariate or multivariate (i.e., which features and number of features)
Outlier detection and removal
Missing value imputation
'''

#X1 = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
#y1 = np.array(housing_df['ASPUS_3A_PCT_CHG'])
import pandas as pd
import numpy as np
h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df2.csv', index_col='DATE', parse_dates=True)
h_m_df.head()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

'''
Plot my data
There does appear to be an overall increasing trend. 
There appears to be some differences in the variance over time. 
There may be some seasonality (i.e., cycles) in the data.
Not sure about outliers.'''

# Plot time series data
f, ax = plt.subplots(1,1)
ax.plot(h_m_df['ASPUS_M'])

# Add title
ax.set_title('Time-series graph for Monly time-series for Average US Housing Price')

# Rotate x-labels
ax.tick_params(axis = 'x', rotation = 45)

# Show graph
plt.show()
plt.close()

'''

Look at stationarity

Most time-series models assume that the underlying time-series data is stationary. This assumption gives us some nice statistical properties that allows us to use various models for forecasting.
Stationarity is a statistical assumption that a time-series has:

    Constant mean
    Constant variance
    Autocovariance does not depend on time
More simply put, if we are using past data to predict future data, we should assume that the data will follow the same general trends and patterns as in the past. This general statement holds for most training data and modeling tasks.

Sometimes we need to transform the data in order to make it stationary. However, this transformation then calls into question if this data is truly stationary and is suited to be modeled using these techniques.

Looking at our data:
Rolling mean and standard deviation look like they change over time. There may be some de-trending and removing seasonality involved. Based on Dickey-Fuller test, because p = 0.31, we fail to reject the null hypothesis (that the time series is not stationary) at the p = 0.05 level, thus concluding that we fail to reject the null hypothesis that our time series is not stationary.
'''
#h_m_df['ASPUS_M']

from statsmodels.tsa.stattools import adfuller

def test_stationarity(df, ts):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    
    # Determing rolling statistics
    rolmean = df[ts].rolling(window = 12, center = False).mean()
    rolstd = df[ts].rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    orig = plt.plot(df[ts], 
                    color = 'blue', 
                    label = 'Original')
    mean = plt.plot(rolmean, 
                    color = 'red', 
                    label = 'Rolling Mean')
    std = plt.plot(rolstd, 
                   color = 'black', 
                   label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df[ts], 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


test_stationarity(df=h_m_df, ts='ASPUS_M')


'''

Correct for stationarity

It is common for time series data to have to correct for non-stationarity. 
2 common reasons behind non-stationarity are:
    Trend – mean is not constant over time.
    Seasonality – variance is not constant over time.
There are ways to correct for trend and seasonality, to make the time series stationary.

What happens if you do not correct for these things?
Many things can happen, including:
Variance can be mis-specified
Model fit can be worse. 
Not leveraging valuable time-dependent nature of the data. 


Eliminating trend and seasonality
Transformation
    Examples. Log, square root, etc.
    We are going to look at log.
Smoothing
    Examples. Weekly average, monthly average, rolling averages.
We are going to look at Monthly & Yearly average.
Differencing
    Examples. First-order differencing.
    We are going to look at first-order differencing.
Polynomial Fitting
    Examples. Fit a regression model.
Decomposition

Transformation, Smoothing, and Differencing
Looking at our data:
Applying log transformation, weekly moving average smoothing, and differencing made the data more stationary over time. Based on Dickey-Fuller test, because p = < 0.05, we fail to reject the null hypothesis (that the time series is not stationary) at the p = 0.05 level, thus concluding that the time series is stationary.'''


def plot_transformed_data(df, ts, ts_transform):
  """
  Plot transformed and original time series data
  """
  # Plot time series data
  f, ax = plt.subplots(1,1)
  ax.plot(df[ts])
  ax.plot(df[ts_transform], color = 'red')

  # Add title
  ax.set_title('%s and %s time-series graph' %(ts, ts_transform))

  # Rotate x-labels
  ax.tick_params(axis = 'x', rotation = 45)

  # Add legend
  ax.legend([ts, ts_transform])
  
  plt.show()
  plt.close()
  
  return



#df_example=h_m_df
#
#ts=ASPUS_M
h_m_df['ASPUS_M']
# Transformation - log ASPUS_M
h_m_df['ASPUS_M_log'] = h_m_df['ASPUS_M'].apply(lambda x: np.log(x))

# Transformation - 7-day moving averages of log ASPUS_M
h_m_df['ASPUS_M_log_moving_avg'] = h_m_df['ASPUS_M_log'].rolling(window = 7,
                                                               center = False).mean()

# Transformation - 7-day moving average ASPUS_M
h_m_df['ASPUS_M_moving_avg'] = h_m_df['ASPUS_M'].rolling(window = 7,
                                                       center = False).mean()

# Transformation - Difference between logged ASPUS_M and first-order difference logged ASPUS_M
# h_m_df['ASPUS_M_log_diff'] = h_m_df['ASPUS_M_log'] - h_m_df['ASPUS_M_log'].shift()
h_m_df['ASPUS_M_log_diff'] = h_m_df['ASPUS_M_log'].diff()

# Transformation - Difference between ASPUS_M and moving average ASPUS_M
h_m_df['ASPUS_M_moving_avg_diff'] = h_m_df['ASPUS_M'] - h_m_df['ASPUS_M_moving_avg']

# Transformation - Difference between logged ASPUS_M and logged moving average ASPUS_M
h_m_df['ASPUS_M_log_moving_avg_diff'] = h_m_df['ASPUS_M_log'] - h_m_df['ASPUS_M_log_moving_avg']

# Transformation - Difference between logged ASPUS_M and logged moving average ASPUS_M
h_m_df_transform = h_m_df.dropna()

# Transformation - Logged exponentially weighted moving averages (EWMA) ASPUS_M
h_m_df_transform['ASPUS_M_log_ewma'] = h_m_df_transform['ASPUS_M_log'].ewm(halflife = 7,
                                                                         ignore_na = False,
                                                                         min_periods = 0,
                                                                         adjust = True).mean()

# Transformation - Difference between logged ASPUS_M and logged EWMA ASPUS_M
h_m_df_transform['ASPUS_M_log_ewma_diff'] = h_m_df_transform['ASPUS_M_log'] - h_m_df_transform['ASPUS_M_log_ewma']

# Display data
display(h_m_df_transform.head())

# Plot data
plot_transformed_data(df = h_m_df, 
                      ts = 'ASPUS_M', 
                      ts_transform = 'ASPUS_M_log')
# Plot data
plot_transformed_data(df = h_m_df, 
                      ts = 'ASPUS_M_log', 
                      ts_transform = 'ASPUS_M_log_moving_avg')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M', 
                      ts_transform = 'ASPUS_M_moving_avg')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M_log', 
                      ts_transform = 'ASPUS_M_log_diff')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M', 
                      ts_transform = 'ASPUS_M_moving_avg_diff')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M_log', 
                      ts_transform = 'ASPUS_M_log_moving_avg_diff')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M_log', 
                      ts_transform = 'ASPUS_M_log_ewma')

# Plot data
plot_transformed_data(df = h_m_df_transform, 
                      ts = 'ASPUS_M_log', 
                      ts_transform = 'ASPUS_M_log_ewma_diff')

# Perform stationarity test
h_m_df['ASPUS_M_log']
test_stationarity(df = h_m_df_transform, 
                  ts = 'ASPUS_M_log')


# Perform stationarity test
test_stationarity(df = h_m_df_transform, 
                  ts = 'ASPUS_M_moving_avg')

# Perform stationarity test
test_stationarity(df = h_m_df_transform, 
                  ts = 'ASPUS_M_log_moving_avg')

# Perform stationarity test
test_stationarity(df = h_m_df_transform,
                  ts = 'ASPUS_M_log_diff')
#p-value                          0.001301

# Perform stationarity test
test_stationarity(df = h_m_df_transform,
                  ts = 'ASPUS_M_moving_avg_diff')
#p-value                          0.002171

# Perform stationarity test
test_stationarity(df = h_m_df_transform,
                  ts = 'ASPUS_M_log_moving_avg_diff')
#p-value                         0.000806  BEST

# Perform stationarity test
test_stationarity(df = h_m_df_transform, 
                  ts = 'ASPUS_M_log_ewma')

# Perform stationarity test
test_stationarity(df = h_m_df_transform,
                  ts ='ASPUS_M_log_ewma_diff')
#p-value                          0.008079

'''
Decomposition: trend, seasonality, residuals¶
Looking at our data:
De-trending and de-seasonalizing made the data (i.e., the residuals) more stationary over time. Based on Dickey-Fuller test, because p = < 0.05, we fail to reject the null hypothesis (that the time series is not stationary) at the p = 0.05 level, thus concluding that the time series is stationary.'''

def plot_decomposition(df, ts, trend, seasonal, residual):
  """
  Plot time series data
  """
  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15, 5), sharex = True)

  ax1.plot(df[ts], label = 'Original')
  ax1.legend(loc = 'best')
  ax1.tick_params(axis = 'x', rotation = 45)

  ax2.plot(df[trend], label = 'Trend')
  ax2.legend(loc = 'best')
  ax2.tick_params(axis = 'x', rotation = 45)

  ax3.plot(df[seasonal],label = 'Seasonality')
  ax3.legend(loc = 'best')
  ax3.tick_params(axis = 'x', rotation = 45)

  ax4.plot(df[residual], label = 'Residuals')
  ax4.legend(loc = 'best')
  ax4.tick_params(axis = 'x', rotation = 45)
  plt.tight_layout()

  # Show graph
  plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 
               x = 0.5, 
               y = 1.05, 
               fontsize = 18)
  plt.show()
  plt.close()
  
  return

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(h_m_df_transform['ASPUS_M_log_moving_avg_diff'], freq = 12)

h_m_df_transform.loc[:,'trend'] = decomposition.trend
h_m_df_transform.loc[:,'seasonal'] = decomposition.seasonal
h_m_df_transform.loc[:,'residual'] = decomposition.resid

plot_decomposition(df = h_m_df_transform, 
                   ts = 'ASPUS_M_log_moving_avg_diff', 
                   trend = 'trend',
                   seasonal = 'seasonal', 
                   residual = 'residual')

test_stationarity(df = h_m_df_transform.dropna(), ts = 'residual')

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(h_m_df_transform['ASPUS_M_log_diff'], freq = 12)

h_m_df_transform.loc[:,'trend'] = decomposition.trend
h_m_df_transform.loc[:,'seasonal'] = decomposition.seasonal
h_m_df_transform.loc[:,'residual'] = decomposition.resid

plot_decomposition(df = h_m_df_transform, 
                   ts = 'ASPUS_M_log_diff', 
                   trend = 'trend',
                   seasonal = 'seasonal', 
                   residual = 'residual')

test_stationarity(df = h_m_df_transform.dropna(), ts = 'residual')

'''
Let us model some time-series data! Finally! ARIMA models.

We will be doing an example here! We can use ARIMA models when we know there is dependence between values and we can leverage that information to forecast.
ARIMA = Auto-Regressive Integrated Moving Average.
Assumptions. The time-series is stationary.
Depends on:
1. Number of AR (Auto-Regressive) terms (p).
2. Number of I (Integrated or Difference) terms (d).
3. Number of MA (Moving Average) terms (q). 

ACF and PACF Plots
How do we determine p, d, and q? For p and q, we can use ACF and PACF plots (below).
Autocorrelation Function (ACF). Correlation between the time series with a lagged version of itself (e.g., correlation of Y(t) with Y(t-1)).
Partial Autocorrelation Function (PACF). Additional correlation explained by each successive lagged term.
How do we interpret ACF and PACF plots?
p – Lag value where the PACF chart crosses the upper confidence interval for the first time.
q – Lag value where the ACF chart crosses the upper confidence interval for the first time.

'''
def plot_acf_pacf(df, ts):
  """
  Plot auto-correlation function (ACF) and partial auto-correlation (PACF) plots
  """
  f, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5)) 

  #Plot ACF: 

  ax1.plot(lag_acf)
  ax1.axhline(y=0,linestyle='--',color='gray')
  ax1.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
  ax1.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
  ax1.set_title('Autocorrelation Function for %s' %(ts))

  #Plot PACF:
  ax2.plot(lag_pacf)
  ax2.axhline(y=0,linestyle='--',color='gray')
  ax2.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
  ax2.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
  ax2.set_title('Partial Autocorrelation Function for %s' %(ts))
  
  plt.tight_layout()
  plt.show()
  plt.close()
  
  return

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

# determine ACF and PACF
lag_acf = acf(np.array(h_m_df_transform['ASPUS_M_log_moving_avg_diff']), nlags = 60)
lag_pacf = pacf(np.array(h_m_df_transform['ASPUS_M_log_moving_avg_diff']), nlags = 60)

# plot ACF and PACF
plot_acf_pacf(df = h_m_df_transform, ts = 'ASPUS_M_log_moving_avg_diff')


from statsmodels.tsa.stattools import acf, pacf

# determine ACF and PACF
lag_acf = acf(np.array(h_m_df_transform['ASPUS_M_log_diff']), nlags = 60)
lag_pacf = pacf(np.array(h_m_df_transform['ASPUS_M_log_diff']), nlags = 60)

# plot ACF and PACF
plot_acf_pacf(df = h_m_df_transform, ts = 'ASPUS_M_log_diff')


#unc arima
def run_arima_model(df, ts, p, d, q):
  """
  Run ARIMA model
  """
  from statsmodels.tsa.arima_model import ARIMA

  # fit ARIMA model on time series
  model = ARIMA(df[ts], order=(p, d, q))  
  results_ = model.fit(disp=-1)  
  
  # get lengths correct to calculate RSS
  len_results = len(results_.fittedvalues)
  ts_modified = df[ts][-len_results:]
  
  # calculate root mean square error (RMSE) and residual sum of squares (RSS)
  rss = sum((results_.fittedvalues - ts_modified)**2)
  rmse = np.sqrt(rss / len(df[ts]))
  
  # plot fit
  plt.plot(df[ts])
  plt.plot(results_.fittedvalues, color = 'red')
  plt.title('For ARIMA model (%i, %i, %i) for ts %s, RSS: %.4f, RMSE: %.4f' %(p, d, q, ts, rss, rmse))
  
  plt.show()
  #plt.close()
  
  return results_


# Note: I do the differencing in the transformation of the data 'ts_log_diff'
# AR model with 1st order differencing - ARIMA (1,0,0)
model_AR_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_moving_avg_diff', 
                           p = 1, 
                           d = 0, 
                           q = 0)

# MA model with 1st order differencing - ARIMA (0,0,1)
model_MA_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_moving_avg_diff', 
                           p = 0, 
                           d = 0, 
                           q = 1)

# ARMA model with 1st order differencing - ARIMA (1,0,1)
model_MA_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_moving_avg_diff', 
                           p = 1, 
                           d = 0, 
                           q = 1)

########
model_AR_log_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 1, 
                           d = 0, 
                           q = 0)

# MA model with 1st order differencing - ARIMA (0,0,1)
model_AR_log_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 0, 
                           d = 0, 
                           q = 1)

# ARMA model with 1st order differencing - ARIMA (1,0,1) BEST
model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 1, 
                           d = 0, 
                           q = 1)

model_AR_log_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 1, 
                           q = 7)

# MA model with 1st order differencing - ARIMA (0,0,1)
model_AR_log_diff = run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 1, 
                           q = 9)

# ARMA model with 1st order differencing - ARIMA (1,0,1) BEST
model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 1, 
                           q = 11)

model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 0, 
                           q = 11)

model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 0, 
                           q = 6)

model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 2, 
                           d = 1, 
                           q = 6)

model_AR_log_diff= run_arima_model(df = h_m_df_transform, 
                           ts = 'ASPUS_M_log_diff', 
                           p = 3, 
                           d = 0, 
                           q = 6)

# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

ASPUS_M_moving_avg_diff=h_m_df_transform['ASPUS_M_log_moving_avg_diff']
ASPUS_M_moving_avg_diff = ARMA(ASPUS_M_moving_avg_diff, order=(1,0, 1))

ASPUS_M_moving_avg_diff = ASPUS_M_moving_avg_diff.fit()
print("The AIC for an AR(1) is: ", ASPUS_M_moving_avg_diff.aic)

ASPUS_M_moving_avg_diff.plot_predict(start='12-01-01', end='12-01-2024')

# Plot the original series and the forecasted series
model_MA_diff.plot_predict(start='12-01-1998', end='12-01-2024')


plt.legend(fontsize=10)
plt.title('Mmodel_MA_diff Forecast')
plt.show()


#####

import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

ASPUS_M_log_diff=h_m_df_transform['ASPUS_M_log_diff']
ASPUS_M_log_diff = ARMA(ASPUS_M_log_diff, order=(1,0, 1))

ASPUS_M_log_diff = ASPUS_M_log_diff.fit()
print("The AIC for an AR(1) is: ", ASPUS_M_log_diff.aic)
print("The BIC for an AR(1) is: ", ASPUS_M_log_diff.bic)

h_m_df['ASPUS_M_log_diff_rescale'] = np.exp(h_m_df['ASPUS_M_log_diff'] + h_m_df['ASPUS_M_log'])
h_m_df['ASPUS_M_log_diff_rescale']=h_m_df['ASPUS_M_log_diff_rescale'].fillna(method='bfill')
#viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])

ASPUS_M_log_diff.plot_predict(start='12-01-98', end='12-01-2024')
h_m_df[['ASPUS_M', 'ASPUS_M_log_diff_rescale']].plot()
#ASPUS_M_log_diff_rescale.plot_predict(start='12-01-01', end='12-01-2024')
X3= h_m_df['ASPUS_M_log_diff_rescale']
X3 = ARMA(X3, order=(3,0,6))  #Good

X3=X3.fit()
X3.plot_predict(start='12-01-98', end='12-01-2024')



# Plot the original series and the forecasted series
#model_MA_diff.plot_predict(start='12-01-1998', end='12-01-2024')


plt.legend(fontsize=10)
plt.title('ASPUS_M_log_diff Forecast')
plt.show()


########
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

ASPUS_M_log_diff=h_m_df_transform['ASPUS_M_log_diff']
#ASPUS_M_log_diff = ARMA(ASPUS_M_log_diff, order=(1,0, 1))
#
#ASPUS_M_log_diff = ASPUS_M_log_diff.fit()

h_m_df['ASPUS_M_log_diff_rescale'] = np.exp(h_m_df['ASPUS_M_log_diff'] + h_m_df['ASPUS_M_log'])
h_m_df['ASPUS_M_log_diff_rescale']=h_m_df['ASPUS_M_log_diff_rescale'].fillna(method='bfill')
#viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])

#ASPUS_M_log_diff.plot_predict(start='12-01-01', end='12-01-2024')
#
#ASPUS_M_log_diff_rescale.plot_predict(start='12-01-01', end='12-01-2024')
X3= h_m_df['ASPUS_M_log_diff_rescale']
#X3 = ARMA(X3, order=(1,0, 1))
#
#X3=X3.fit()

X3 = sm.tsa.ARIMA(X3, (2, 1, 9)).fit()
predictions = X3.forecast()[0]
print(predictions)
X3.plot_predict(start='12-01-98', end='12-01-2024')

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.80)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = sqrt(mean_squared_error(test, predictions))
	return error
#X=X3.values
X = h_m_df['ASPUS_M_log_diff_rescale'].values
#r2 = r2_score(X, predictions)

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
                
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue

	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# evaluate parameters
p_values = [0, 1, 2]
d_values = range(0, 3)
q_values = range(0, 3)
import warnings
warnings.filterwarnings('ignore')

evaluate_models(X, p_values, d_values, q_values)

'''
p_values = [1, 2, 3]
d_values = range(1, 3)
q_values = range(6, 8)
warnings.filterwarnings("ignore")
evaluate_models(X, p_values, d_values, q_values)
ARIMA(1, 1, 6) RMSE=4805.848
ARIMA(1, 1, 7) RMSE=4712.627
ARIMA(1, 2, 6) RMSE=4848.832
ARIMA(1, 2, 7) RMSE=4806.178
ARIMA(2, 1, 6) RMSE=4688.490
ARIMA(2, 1, 7) RMSE=4738.802
ARIMA(2, 2, 6) RMSE=4812.737
ARIMA(2, 2, 7) RMSE=4865.924
ARIMA(3, 1, 6) RMSE=4808.036
ARIMA(3, 1, 7) RMSE=4797.000
ARIMA(3, 2, 7) RMSE=4880.132
Best ARIMA(2, 1, 6) RMSE=4688.490'''

evaluate_arima_model(X, (1,1,2))
evaluate_arima_model(X, (2,1,1))
# evaluate forecasts
print("The AIC for an AR(1) is: ", X3.aic)
print("The BIC for an AR(1) is: ", X3.bic)
print(X3.summary())

fig, ax = plt.subplots()
X3.plot_predict(start='12-01-1998', end='12-01-2023', ax=ax, dynamic=False, plot_insample=True)

h_m_df[['ASPUS_M', 'ASPUS_M_log_diff_rescale']].plot()

# Plot the original series and the forecasted series
#model_MA_diff.plot_predict(start='12-01-1998', end='12-01-2024')

#X3.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
sns.despine()
plt.legend(fontsize=10)
plt.title('ASPUS_M_log_diff Forecast')
plt.show()

X4 = evaluate_arima_model(X, (2,1,6))
print("The AIC for an AR(1) is: ", X4.aic)
print("The BIC for an AR(1) is: ", X4.bic)
print(X4.summary())

#########  SARIMAX


from __future__ import absolute_import, division, print_function
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline
%config InlineBackend.figure_format='retina'



import sys
import os

import pandas as pd
import numpy as np

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')


###

import pandas as pd
import numpy as np
h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df2.csv', index_col='DATE', parse_dates=True)
h_m_df.head()
X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])

h_m_df['ASPUS_M_log'] = h_m_df['ASPUS_M'].apply(lambda x: np.log(x))

h_m_df['ASPUS_M_log_diff'] = h_m_df['ASPUS_M_log'].diff()

ASPUS_M_log_diff

h_m_df['ASPUS_M_log_diff_rescale'] = np.exp(h_m_df['ASPUS_M_log_diff'] + h_m_df['ASPUS_M_log'])
h_m_df['ASPUS_M_log_diff_rescale']=h_m_df['ASPUS_M_log_diff_rescale'].fillna(method='bfill')

ts_df= h_m_df['ASPUS_M_log_diff_rescale']

n_sample = ts_df.shape[0]

print(ts_df.shape)
print(n_sample)
print(ts_df.head())

###

# Create a training sample and testing sample before analyzing the series

n_train=int(0.95*n_sample)+1
n_forecast=n_sample-n_train
#ts_df
ts_train = h_m_df.iloc[:n_train]['ASPUS_M_log_diff_rescale']
ts_test = h_m_df.iloc[n_train:]['ASPUS_M_log_diff_rescale']
print(ts_train.shape)
print(ts_test.shape)
print("Training Series Tail:", "\n", ts_train.tail(), "\n")
print("Testing Series Head:", "\n", ts_test.head())


######

def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.
    
    Source: https://tomaugspurger.github.io/modern-7-timeseries.html
    '''
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax


tsplot(ts_train, title='A Given Training Series', lags=60);


'''** Observations from the sample ACF and sample PACF (based on first 60 lags) **
The sample autocorrelation gradually tails off.
The sample partial autocorrelation does not exactly cut off at some lag p but does not exactly tail off either.
Based on these observations, we could attempt an ARIMA(1,1,1) model as a starting point, although other orders could serve as candidates as well.'''

#Model Estimation

# Fit the model
arima111 = sm.tsa.SARIMAX(ts_train, order=(1,1,1))
model_results = arima111.fit()
model_results.summary()

'''
Digression:
In practice, one could search over a few models using the visual clues above as a starting point. 
The code below gives one such example'''

import itertools

p_min = 0
d_min = 0
q_min = 0
p_max = 4
d_max = 0
q_max = 4

# Initialize a DataFrame to store the results
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    
    try:
        model = sm.tsa.SARIMAX(ts_train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

###

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 );
ax.set_title('BIC');

# Alternative model selection method, limited to only searching AR and MA parameters

train_results = sm.tsa.arma_order_select_ic(ts_train, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

'''3.2 Model Diagnostic Checking
Conduct visual inspection of the residual plots
Residuals of a well-specified ARIMA model should mimic Gaussian white noises: the residuals should be uncorrelated and distributed approximated normally with mean zero and variance 
n^−1
Apparent patterns in the standardized residuals and the estimated ACF of the residuals give an indication that the model need to be re-specified
The results.plot_diagnostics() function conveniently produce several plots to facilitate the investigation.
The estimation results also come with some statistical tests'''

# Residual Diagnostics
# The plot_diagnostics function associated with the estimated result object produce a few plots that allow us 
# to examine the distribution and correlation of the estimated residuals

model_results.plot_diagnostics(figsize=(16, 12));

'''3.2.1 Formal testing
** More information about the statistics under the parameters table, tests of standardized residuals **
Test of heteroskedasticity
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_heteroskedasticity
Test of normality (Jarque-Bera)
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_normality
Test of serial correlation (Ljung-Box)
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_serial_correlation.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.test_serial_correlation'''

# Re-run the above statistical tests, and more. To be used when selecting viable models.

het_method='breakvar'
norm_method='jarquebera'
sercor_method='ljungbox'

(het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[0]
sercor_stat, sercor_p = model_results.test_serial_correlation(method=sercor_method)[0]
sercor_stat = sercor_stat[-1] # last number for the largest lag
sercor_p = sercor_p[-1] # last number for the largest lag

# Run Durbin-Watson test on the standardized residuals.
# The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
# Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
# This statistic will always be between 0 and 4. The closer to 0 the statistic,
# the more evidence for positive serial correlation. The closer to 4,
# the more evidence for negative serial correlation.
# Essentially, below 1 or above 3 is bad.
dw = sm.stats.stattools.durbin_watson(model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

# check whether roots are outside the unit circle (we want them to be);
# will be True when AR is not used (i.e., AR order = 0)
arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
# will be True when MA is not used (i.e., MA order = 0)
maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)

print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(het_method, het_stat, het_p));
print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(norm_method, norm_stat, norm_p));
print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(sercor_method, sercor_stat, sercor_p));
print('\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw))
print('\nTest for all AR roots outside unit circle (>1): {}'.format(arroots_outside_unit_circle))
print('\nTest for all MA roots outside unit circle (>1): {}'.format(maroots_outside_unit_circle))

'''3.3 Model performance evaluation (in-sample fit)'''

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
ax1.plot(ts_train, label='In-sample data', linestyle='-')
# subtract 1 only to connect it to previous point in the graph
ax1.plot(ts_test, label='Held-out data', linestyle='--')

# yes DatetimeIndex
pred_begin = ts_train.index[model_results.loglikelihood_burn]
pred_end = ts_test.index[-1]
pred = model_results.get_prediction(start=pred_begin.strftime('%Y-%m-%d'),
                                    end=pred_end.strftime('%Y-%m-%d'))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int(alpha=0.05)

ax1.plot(pred_mean, 'r', alpha=.6, label='Predicted values')
ax1.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=.2)

ax1.legend(loc='best');

###############
def get_rmse(y, y_hat):
    '''Root Mean Square Error
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def get_mape(y, y_hat):
    '''Mean Absolute Percent Error
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    '''
    perc_err = (100*(y - y_hat))/y
    return np.mean(abs(perc_err))

def get_mase(y, y_hat):
    '''Mean Absolute Scaled Error
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    '''
    abs_err = abs(y - y_hat)
    dsum=sum(abs(y[1:] - y_hat[1:]))
    t = len(y)
    denom = (1/(t - 1))* dsum
    return np.mean(abs_err/denom)

'''3.4 Forecasting and forecast evaluation'''
rmse = get_rmse(ts_train, pred_mean.ix[ts_train.index])
print("RMSE: ", rmse)

mape = get_mape(ts_train, pred_mean.ix[ts_train.index])
print("MAPE: ", mape)

mase = get_mase(ts_train, pred_mean.ix[ts_train.index])
print("MASE: ", mase)
#####

#############

tsplot(ts_df, title='US AVE Housing Price, 1998-2018', lags=60);


'''
4.2 Building a Seasonal ARIMA Model for Forecasting
'''
h_m_df.info()
h_m_df['ASPUS_M_log'] = h_m_df['ASPUS_M'].apply(lambda x: np.log(x))

h_m_df['ASPUS_M_log_diff'] = h_m_df['ASPUS_M_log'].diff()

h_m_df['ASPUS_M_log_diff_rescale'] = np.exp(h_m_df['ASPUS_M_log_diff'] + h_m_df['ASPUS_M_log'])
h_m_df['ASPUS_M_log_diff_rescale']=h_m_df['ASPUS_M_log_diff_rescale'].fillna(method='bfill')

X3= h_m_df['ASPUS_M_log_diff_rescale']

tsplot(h_m_df['ASPUS_M_log'], title='Natural Log of US AVE Housing Price, 1998-2018', lags=40);

tsplot(h_m_df['ASPUS_M_log_diff'], title='Difference in Log of Housing Price, 1998-2018', lags=40);

######### Box Plot for seasonality
h_m_df['Month'] = h_m_df.index.strftime('%b')
h_m_df['Year'] = h_m_df.index.year

h_m_df_piv = h_m_df.pivot(index='Year', columns='Month', values='ASPUS_M')

h_m_df = h_m_df.drop(['Month', 'Year'], axis=1)

# put the months in order
month_names = pd.date_range(start='2000-01-01', periods=12, freq='MS').strftime('%b')
h_m_df_piv = h_m_df_piv.reindex(columns=month_names)

# plot it
fig, ax = plt.subplots(figsize=(8, 6))
h_m_df_piv.plot(ax=ax, kind='box');

ax.set_xlabel('Month');
ax.set_ylabel('Housing Price ($)');
ax.set_title('Boxplot of seasonal values');
plt.xticks(rotation = 45)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout();

###############

# Housing Average Price Monthly Series
mod = sm.tsa.statespace.SARIMAX(ts_train, order=(2,1,0), seasonal_order=(1,1,0,12), )
sarima_fit1 = mod.fit()
print(sarima_fit1.summary())


mod = sm.tsa.statespace.SARIMAX(ts_train, order=(0,1,1), seasonal_order=(0,1,0,12))
sarima_fit2 = mod.fit()
print(sarima_fit2.summary())


'''Notice an additional argument simple_differencing=True. 
This controls how the order of integration is handled in ARIMA models. 
If simple_differencing=True, then the time series provided as endog is literally differenced and an ARMA model is fit to the resulting new time series. This implies that a number of initial periods are lost to the differencing process, however it may be necessary either to compare results to other packages (e.g. Stata's arima always uses simple differencing) or if the seasonal periodicity is large'''
# Model Diagnostic

sarima_fit1.plot_diagnostics(figsize=(16, 12));

sarima_fit2.plot_diagnostics(figsize=(16, 12));
################

n_train=int(0.95*n_sample)+1
n_forecast=n_sample-n_train
#ts_df
ts_train = h_m_df.iloc[:n_train]['ASPUS_M_log_diff_rescale']
ts_test = h_m_df.iloc[n_train:]['ASPUS_M_log_diff_rescale']
print(ts_train.shape)
print(ts_test.shape)
print("Training Series Tail:", "\n", ts_train.tail(), "\n")
print("Testing Series Head:", "\n", ts_test.head())


# Step 5: Do a 14-step ahead forecast

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    
ax1.plot(ts_train, label='In-sample data', linestyle='-')
# subtract 1 only to connect it to previous point in the graph
ax1.plot(ts_test, label='Held-out data', linestyle='--')

# yes DatetimeIndex
pred_begin = ts_train.index[sarima_fit2.loglikelihood_burn]
pred_end = ts_test.index[-1]
pred = sarima_fit2.get_prediction(start=pred_begin.strftime('%Y-%m-%d'),
                                    end=pred_end.strftime('%Y-%m-%d'))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int(alpha=0.05)

ax1.plot(pred_mean, 'r', alpha=.6, label='Predicted values')
ax1.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=.2)
ax1.set_xlabel("Year")
ax1.set_ylabel("Housing Price ($) ")
ax1.legend(loc='best');
ax.set_title('US AVE Housing Price Since 1962');
plt.xticks(rotation = 45)
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout();

## Discuss the results.  How does your forecast look?


                                                                                                                                                                    
#AR MODEL

ts_log= h_m_df.iloc[:n_train]['ASPUS_M_log']
ts_log_diff= h_m_df.iloc[:n_train]['ASPUS_M_log_diff']

model = ARIMA(ts_log, order=(2,1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

#MA Model
model = ARIMA(ts_log, order=(1, 1, 6))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

#Combined Model
model = ARIMA(ts_log, order=(2, 1, 6))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


######

'''Taking it back to original scale
Since the combined model gave best result, lets scale it back to the original values and see how well it performs there. First step would be to store the predicted results as a separate series and observe it.'''

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                    
'''first determine the cumulative sum at index and then add it to the base number. The cumulative sum can be found as:'''

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())
print (predictions_ARIMA_diff_cumsum.tail())

'''Next we’ve to add them to base number. For this lets create a series with all values as base number and add the differences to it. This can be done as:'''

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

'''
Here the first element is base number itself and from thereon the values cumulatively added. Last step is to take the exponent and compare with the original series.'''
ts=h_m_df['ASPUS_M']
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.legend(fontsize=10)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

#Comparing last meathod

h_m_df['ASPUS_M_log_diff_rescale'] = np.exp(h_m_df['ASPUS_M_log_diff'] + h_m_df['ASPUS_M_log'])
h_m_df['ASPUS_M_log_diff_rescale']=h_m_df['ASPUS_M_log_diff_rescale'].fillna(method='bfill')

X3= h_m_df['ASPUS_M_log_diff_rescale']

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(X3)
plt.plot(predictions_ARIMA)
plt.legend(fontsize=10)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(X3)))



#############  ARIMA
X3= h_m_df['ASPUS_M_log_diff_rescale']

X3 = sm.tsa.ARIMA(X3, (3, 1, 6)).fit()
predictions = X3.forecast()[0]
print(predictions)
#X3.plot_predict(start='12-01-98', end='12-01-2024')
fig, ax = plt.subplots()
X3.plot_predict(start='12-01-1998', end='12-01-2024', ax=ax, dynamic=False, plot_insample=True)

X2 = predictions_ARIMA
X2 = sm.tsa.ARIMA(X2, (3, 1, 6)).fit()
predictions = X2.forecast()[0]
print(predictions)
#X3.plot_predict(start='12-01-98', end='12-01-2024')
fig, ax = plt.subplots()
X2.plot_predict(start='12-01-1998', end='12-01-2024', ax=ax, dynamic=False, plot_insample=True)


#X3.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
sns.despine()
plt.legend(fontsize=10)
plt.title('ASPUS_M_log_diff Forecast')
plt.show()

#########  ARMA

X3= h_m_df['ASPUS_M_log_diff_rescale']

X3 = sm.tsa.ARMA(X3, (3, 0, 6)).fit()
predictions = X3.forecast()[0]
print(predictions)
#X3.plot_predict(start='12-01-98', end='12-01-2024')
fig, ax = plt.subplots()
X3.plot_predict(start='12-01-1998', end='12-01-2024', ax=ax, dynamic=False, plot_insample=True)

X2 = predictions_ARIMA
X2 = sm.tsa.ARMA(X2, (3, 0, 6)).fit()
predictions = X2.forecast()[0]
print(predictions)
#X3.plot_predict(start='12-01-98', end='12-01-2024')
fig, ax = plt.subplots()
X2.plot_predict(start='12-01-1998', end='12-01-2024', ax=ax, dynamic=False, plot_insample=True)


#X3.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
sns.despine()
plt.legend(fontsize=10)
plt.title('ASPUS_M_log_diff Forecast')
plt.show()


#########
'''  FBProphet '''

'''

Let us model some time-series data! Finally! Facebook Prophet package.
We will be doing an example here! Installing the necessary packages might take a couple of minutes. In the meantime, I can talk a bit about Facebook Prophet, a tool that allows folks to forecast using additive or component models relatively easily. It can also include things like:
Day of week effects
Day of year effects
Holiday effects
Trend trajectory
Can do MCMC sampling

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Accurate and fast.
Prophet is used in many applications across Facebook for producing reliable forecasts for planning and goal setting. We’ve found it to perform better than any other approach in the majority of cases. We fit models in Stan so that you get forecasts in just a few seconds.

Get a reasonable forecast on messy data with no manual effort. Prophet is robust to outliers, missing data, and dramatic changes in your time series.

Tunable forecasts.
The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human-interpretable parameters to improve your forecast by adding your domain knowledge.

Prophet follows the sklearn model API. We create an instance of the Prophet class and then call its fit and predict methods.
The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

https://facebook.github.io/prophet/docs/quick_start.html#python-api

We fit the model by instantiating a new Prophet object. Any settings to the forecasting procedure are passed into the constructor. Then you call its fit method and pass in the historical dataframe. Fitting should take 1-5 seconds.

# Python
m = Prophet()
m.fit(df)

Predictions are then made on a dataframe with a column ds containing the dates for which a prediction is to be made. You can get a suitable dataframe that extends into the future a specified number of days using the helper method Prophet.make_future_dataframe. By default it will also include the dates from the history, so we will see the model fit as well.

Using Prophet is extremely straightforward. You import it, load some data into a pandas dataframe, set the data up into the proper format and then start modeling / forecasting.
'''
housing_df.info()
housing_df = pd.read_csv('C:/scripts/capstone2/housing_df.csv', index_col=0)

h_m_df.info()
h_m_df = pd.read_csv('C:/scripts/capstone2/h_m_df.csv', index_col=0)
h_m_df.head()



#conda install pystan
#!pip install fbprophet
from fbprophet import Prophet
import datetime
from datetime import datetime

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')


X1 = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
y1 = np.array(housing_df['ASPUS_3A_PCT_CHG'])

X = np.array(h_m_df.drop(['ASPUS_M'],1))
y = np.array(h_m_df['ASPUS_M'])


'''Prepare for Prophet
For prophet to work, we need to change the names of these columns to 'ds' and 'y', so lets just create a new dataframe and keep our old one handy (you'll see why later). The new dataframe will initially be created with an integer index so we can rename the columns

                                                                                                                                                  
'''


df = h_m_df.reset_index()
df.head()

'''
Let's rename the columns as required by fbprophet. Additioinally, fbprophet doesn't like the index to be a datetime...it wants to see 'ds' as a non-index column, so we won't set an index differnetly than the integer index.

'''
df=df.rename(columns={'DATE':'ds', 'ASPUS_M':'y'})
df1=df.rename(columns={'DATE':'ds', 'ASPUS_M':'y'})

'''Note the format of the dataframe. This is the format that Prophet expects to see. There needs to be a ‘ds’ column  that contains the datetime field and and a ‘y’ column that contains the value we are wanting to model/forecast.'''

df=df[['ds', 'y']]
df.tail()

'''
Before we can do any analysis with this data, we need to log transform the ‘y’ variable to a try to convert non-stationary data to stationary. This also converts trends to more linear trends (see this website for more info). This isn’t always a perfect way to handle time-series data, but it works often enough that it can be tried initially without much worry.

To log-tranform the data, we can use np.log() on the ‘y’ column like this:
    
'''

df['y_orig'] = df['y'] # to save a copy of the original data..you'll see why shortly. 
# log-transform y
df['y'] = np.log(df['y'])
df.tail()

'''Its time to start the modeling.  You can do this easily with the following command:'''

model = Prophet() #instantiate Prophet
model.fit(df) #fit the model with your dataframe

'''Now its time to start forecasting. With Prophet, you start by building some future time data with the following command:'''
#create 12 months of future data
future_data_y = model.make_future_dataframe(periods=12, freq = 'm')
 
#forecast the data for future data
forecast_data_y = model.predict(future_data_y)

#create 24 months of future data
future_data_2y = model.make_future_dataframe(periods=24, freq = 'm')

#forecast the data for future data
forecast_data_2y = model.predict(future_data_2y)


#create 36 months of future data
future_data_3y = model.make_future_dataframe(periods=36, freq = 'm')

#forecast the data for future data
forecast_data_3y = model.predict(future_data_3y)

future_data_5y = model.make_future_dataframe(periods=60, freq = 'm')

#forecast the data for future data
forecast_data_5y = model.predict(future_data_5y)

future_data_10y = model.make_future_dataframe(periods=120, freq = 'm')

#forecast the data for future data
forecast_data_10y = model.predict(future_data_10y)
forecast_data_3y.tail()
forecast_data_3y.info()

'''In this line of code, we are creating a pandas dataframe with 12 (periods = 12) future data points with a monthly frequency (freq = ‘m’).  If you’re working with daily data, you wouldn’t want include freq=’m’.'''


'''Now we forecast using the ‘predict’ command:
If you take a quick look at the data using .head() or .tail(), you’ll notice there are a lot of columns in the forecast_data dataframe. The important ones (for now) are ‘ds’ (datetime), ‘yhat’ (forecast), ‘yhat_lower’ and ‘yhat_upper’ (uncertainty levels).'''

forecast_data_y[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

'''Let’s take a look at a graph of this data to get an understanding of how well our model is working.'''

model.plot(forecast_data_y)
model.plot(forecast_data_3y)
model.plot(forecast_data_5y)
model.plot(forecast_data_10y)


'''That looks pretty good. Now, let’s take a look at the seasonality and trend components of our /data/model/forecast.'''

model.plot_components(forecast_data_5y)


'''From the trend and seasonality, we can see that the trend is a playing a large part in the underlying time series and seasonality comes into play more toward the beginning of the year.

So far so good.  With the above info, we’ve been able to quickly model and forecast some data to get a feel for what might be coming our way in the future from this particular data set.

Forecast plot to display our ‘original’ data so you can see the forecast in ‘context’ and in the original scale rather than the log-transformed data. We can do this by using np.exp() to get our original data back.'''

forecast_data_5y_orig = forecast_data_5y # make sure we save the original forecast data
forecast_data_5y_orig['yhat'] = np.exp(forecast_data_5y_orig['yhat'])
forecast_data_5y_orig['yhat_lower'] = np.exp(forecast_data_5y_orig['yhat_lower'])
forecast_data_5y_orig['yhat_upper'] = np.exp(forecast_data_5y_orig['yhat_upper'])

'''Let’s take a look at the forecast with the original data:'''
model.plot(forecast_data_5y_orig)

'''Something looks wrong (and it is)!

Our original data is drawn on the forecast but the black dots (the dark line at the bottom of the chart) is our log-transform original ‘y’ data. For this to make any sense, we need to get our original ‘y’ data points plotted on this chart. To do this, we just need to rename our ‘y_orig’ column in the df dataframe to ‘y’ to have the right data plotted. Be careful here…you want to make sure you don’t continue analyzing data with the non-log-transformed data.'''

df['y_log']=df['y'] #copy the log-transformed data to another column
df['y']=df['y_orig'] #copy the original data to 'y'

model.plot(forecast_data_5y_orig)
plt.show()
df.info()

'''There we got a forecast for Housing Price 60 months into the future (you have to look closely at the very far right-hand side for the forecast). It looks like the next sixths months will see sales between 350K and 555K.'''



'''We make a dataframe for future predictions as before, except we must also specify the capacity in the future. Here we keep capacity constant at the same value as in the history, and forecast 3 years into the future:'''


df['cap'] = 30

df['floor'] =1

#considreing your dataframe
#df = pandas.read_csv('yourCSV')
cap = df['cap']
flr = df['floor']
df['cap'] = cap
df['floor'] = flr

df.info()
df.head()
model=Prophet(changepoint_range=0.8, changepoint_prior_scale=0.12, growth='logistic',
                    seasonality_mode='additive', interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True) 
model.add_seasonality(name="monthly", period=30.5, fourier_order=12, prior_scale=0.02)
model.fit(df);

#create 12 months of future data
future_data = model.make_future_dataframe(periods=60, freq = 'm')
type(future_data)
future_data.tail()

future_data['cap']=30
future_data['floor']=1

forecast_data = model.predict(future_data) 
#forecast_data['cap'] = cap
#forecast_data['floor'] =flr
##forecast the data for future data
#forecast_data = model.predict(future_data) 


fig = model.plot(forecast_data, uncertainty=True)

'''If you want to see the forecast components, you can use the Prophet.plot_components method. By default you’ll see the trend, yearly seasonality, and weekly seasonality of the time series. If you include holidays, you’ll see those here, too.

A variation in values from the output presented above is to be expected as Prophet relies on Markov chain Monte Carlo (MCMC) methods to generate its forecasts. MCMC is a stochastic process, so values will be slightly different each time.

Prophet also provides a convenient function to quickly plot the results of our forecasts:
'''

fig2 = model.plot_components(forecast_data, uncertainty=True)

model.plot(forecast_data)

'''While this is a nice chart, it is kind of ‘busy’ for me.  Additionally, I like to view my forecasts with original data first and forecasts appended to the end (this ‘might’ make sense in a minute).

First, we need to get our data combined and indexed appropriately to start plotting. We are only interested (at least for the purposes of this article) in the ‘yhat’, ‘yhat_lower’ and ‘yhat_upper’ columns from the Prophet forecasted dataset.  Note: There are much more pythonic ways to these steps, but I’m breaking them out for each of understanding.'''
df.head()
df.info()
df.set_index('ds', inplace=True)
forecast_data.set_index('ds', inplace=True)
viz_df = df.join(forecast_data[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')

viz_df.info()
viz_df.head()

'''You don’t need to delete the ‘y’and ‘index’ columns, but it makes for a cleaner dataframe.

If you ‘tail’ your dataframe, your data should look something like this:'''

del viz_df['y']
#del viz_df['index']

viz_df=viz_df[['y_orig', 'yhat', 'yhat_lower', 'yhat_upper']]
viz_df.tail()
viz_df.head()
viz_df.describe()

viz_df['yhat_rescaled'] = viz_df['yhat']
viz_df.tail()

'''Let's take a look at the Housing Price and yhat_rescaled data together in a chart.'''


viz_df[['y_orig', 'yhat_rescaled']].plot()


'''First, we need to get the last date in the original sales data. This will be used to split the data for plotting.'''

df.index = pd.to_datetime(df.index)
last_date = df.index[-1]

'''To plot our forecasted data, we’ll set up a function (for re-usability). This function imports a couple of extra libraries for subtracting dates (timedelta) and then sets up the function.'''


from datetime import date,timedelta
 
def plot_data(func_df, end_date):
    end_date = end_date - timedelta(weeks=4) # find the 2nd to last row in the data. We don't take the last row because we want the charted lines to connect
    mask = (func_df.index > end_date) # set up a mask to pull out the predicted rows of data.
    predict_df = func_df.loc[mask] # using the mask, we create a new dataframe with just the predicted data.
   
# Now...plot everything
    fig, ax1 = plt.subplots()
    ax1.plot(sales_df.y_orig)
    ax1.plot((np.exp(predict_df.yhat)), color='black', linestyle=':')
    ax1.fill_between(predict_df.index, np.exp(predict_df['yhat_upper']), np.exp(predict_df['yhat_lower']), alpha=0.5, color='darkgray')
    ax1.set_title('Housing Price Orig (Orange) vs Housing Price Forecast (Black)')
    ax1.set_ylabel('Dollar Sales')
    ax1.set_xlabel('Date')
  
# change the legend text
    L=ax1.legend() #get the legend
    L.get_texts()[0].set_text('Actual Price') #change the legend text for 1st plot
    L.get_texts()[1].set_text('Forecasted Price') #change the legend text for 2nd plo


#Here 032918
#########################


h_m_df2 = pd.read_csv('C:/scripts/capstone2/h_m_df2.csv', index_col='DATE', parse_dates=True)
h_m_df2.head()



#conda install pystan
#!pip install fbprophet
from fbprophet import Prophet
import datetime
from datetime import datetime

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')


#X1 = np.array(housing_df.drop(['ASPUS_3A_PCT_CHG'],1))
#y1 = np.array(housing_df['ASPUS_3A_PCT_CHG'])
#
#X = np.array(h_m_df2.drop(['ASPUS_M'],1))
#y = np.array(h_m_df2['ASPUS_M'])


'''Prepare for Prophet
For prophet to work, we need to change the names of these columns to 'ds' and 'y', so lets just create a new dataframe and keep our old one handy (you'll see why later). The new dataframe will initially be created with an integer index so we can rename the columns

                                                                                                                                                  
'''


df2 = h_m_df2.reset_index()
df3 = h_m_df2.reset_index()  #df3_orig
#df2=[['DATE', 'ASPUS_M']]
df2.head()
df3.head() #orig
df2=df2.rename(columns={'DATE':'ds', 'ASPUS_M':'y'})
df2.head()

'''
Let's rename the columns as required by fbprophet. Additioinally, fbprophet doesn't like the index to be a datetime...it wants to see 'ds' as a non-index column, so we won't set an index differnetly than the integer index.

'''

df2.set_index('ds').y.plot()
df2.head()

'''When working with time-series data, its good to take a look at the data to determine if trends exist, whether it is stationary, has any outliers and/or any other anamolies. Facebook prophet's example uses the log-transform as a way to remove some of these anomolies but it isn't the absolute 'best' way to do this...but given that its the example and a simple data series, I'll follow their lead for now. Taking the log of a number is easily reversible to be able to see your original data.

To log-transform your data, you can use numpy's log() function'''

df2['y_orig'] = df2['y'] # to save a copy of the original data..you'll see why shortly.


df2.tail()

'''
Before we can do any analysis with this data, we need to log transform the ‘y’ variable to a try to convert non-stationary data to stationary. This also converts trends to more linear trends (see this website for more info). This isn’t always a perfect way to handle time-series data, but it works often enough that it can be tried initially without much worry.

To log-tranform the data, we can use np.log() on the ‘y’ column like this:
    
'''
#Linerar Growth
# log-transform y
df2['y'] = np.log(df2['y'])
df2.tail()


df2.set_index('ds').y.plot() 
df2.tail()
'''
As you can see in the above chart, the plot looks the same as 
'''






model = Prophet()
model.fit(df2)

#model = Prophet()#, interval_width=0.95)
#model.fit(df2)

future = model.make_future_dataframe(periods=60, freq = 'm')
future.tail()

df2.head()
forecast = model.predict(future)

forecast.tail()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

'''Plotting Prophet results
Prophet has a plotting mechanism called plot. This plot functionality draws the original data (black dots), the model (blue line) and the error of the forecast (shaded blue area).'''


model.plot(forecast);


'''Visualizing Prophet models
In order to build a useful dataframe to visualize our model versus our original data, we need to combine the output of the Prophet model with our original data set, then we'll build a new chart manually using pandas and matplotlib.

First, let's set our dataframes to have the same index of ds'''

df2.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)

df3.head()
df3.set_index('DATE', inplace=True)
viz_df = df3.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')

'''If we look at the head(), we see the data has been joined correctly but the scales of our original data (sales) and our model (yhat) are different. We need to rescale the yhat colums(s) to get the same scale, so we'll use numpy's exp function to do that.
'''

viz_df.head()


viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
viz_df.head()

'''Let's take a look at the sales and yhat_rescaled data together in a chart.'''


viz_df[['ASPUS_M', 'yhat_rescaled']].plot()

'''You can see from the chart that the model (blue) is pretty good when plotted against the actual signal (orange) but I like to make my vizualization's a little better to understand. To build my 'better' visualization, we'll need to go back to our original df3 and forecast dataframes.

First things first - we need to find the 2nd to last date of the original sales data in sales_df in order to ensure the original sales data and model data charts are connected.
'''

df3.index = pd.to_datetime(df3.index) #make sure our index as a datetime object
connect_date = df3.index[-2] #select the 2nd to last date

'''Using the connect_date we can now grab only the model data that after that date. To do this, we'll mask the forecast data.'''

mask = (forecast.index > connect_date)
predict_df = forecast.loc[mask]

predict_df.info()
predict_df.head()

'''Now, let's build a dataframe to use in our new visualization. We'll follow the same steps we did before.'''

viz_df = df3.join(predict_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
viz_df['yhat_scaled']=np.exp(viz_df['yhat'])

'''Now, if we take a look at the head() of viz_df we'll see 'NaN's everywhere except for our original data rows.
'''
viz_df.head()

'''
If we take a look at the tail() of the viz_df you'll see we have data for the forecasted data and NaN's for the original data series.'''

viz_df.tail()


'''time to plot
Now, let's plot everything to get the 'final' visualization of our sales data and forecast with errors.'''

fig, ax1 = plt.subplots()
ax1.plot(viz_df.ASPUS_M)
ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Housing Price (Orange) vs Hosing Growth Forecast (Black)')
ax1.set_ylabel('Housing Prices($)')
ax1.set_xlabel('Date')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual Housing Prices') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecasted Housing Prices') #change the legend text for 2nd plot


'''
This visualization is much better (in my opinion) than the default fbprophet plot. It is much easier to quickly understand and describe what's happening. The orange line is actual sales data and the black dotted line is the forecast. The gray shaded area is the uncertaintity estimation of the forecast.'''


####################################
import pandas as pd
import numpy as np
h_m_df2 = pd.read_csv('C:/scripts/capstone2/h_m_df2.csv', index_col='DATE', parse_dates=True)
h_m_df2.head()

#conda install pystan
#!pip install fbprophet
from fbprophet import Prophet
import datetime
from datetime import datetime

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

#Looking at the recent data
df4 = h_m_df2.loc['19980102':'20181201'] 
df2 = df4.reset_index()
df3 = df4.reset_index()  #df3_orig
#df2=[['DATE', 'ASPUS_M']]
df2.head()
df3.head() #orig
df2=df2.rename(columns={'DATE':'ds', 'ASPUS_M':'y'})
df2.tail()

'''
Let's rename the columns as required by fbprophet. Additioinally, fbprophet doesn't like the index to be a datetime...it wants to see 'ds' as a non-index column, so we won't set an index differnetly than the integer index.

'''
df2.set_index('ds').y.plot()
df2.head()

df2['y_orig'] = df2['y'] # to save a copy of the original data..you'll see why shortly.

df2.tail()

df2['y'] = np.log(df2['y'])
df2.tail()

df2.set_index('ds').y.plot() 
df2.tail()

### Logistic Growth
df2['cap'] = 30
df2['floor'] =1

#considreing your dataframe
#df = pandas.read_csv('yourCSV')
cap = df2['cap']
flr = df2['floor']
df2['cap'] = cap
df2['floor'] = flr

df2.info()
df2.head()
'''
By default changepoints are only inferred for the first 80% of the time series in order to have plenty of runway for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series. This default works in many situations but not all, and can be change using the changepoint_range argument. For example, m = Prophet(changepoint_range=0.9) in Python will place potential changepoints in the first 90% of the time series.

Adjusting trend flexibility
If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument changepoint_prior_scale. By default, this parameter is set to 0.05. Increasing it will make the trend more flexible:'''
#95-96  .9-.95  -60-80

model=Prophet(changepoint_range=0.97, changepoint_prior_scale=0.96, growth='logistic',
                    seasonality_mode='additive', interval_width=0.70, yearly_seasonality=True, weekly_seasonality=True) 
model.add_seasonality(name="monthly", period=12, fourier_order=120, prior_scale=0.5)
model.fit(df2);

#create 12 months of future data
future_data = model.make_future_dataframe(periods=60, freq = 'm')
type(future_data)
future_data.tail()

future_data['cap']=30
future_data['floor']=1

forecast_data = model.predict(future_data) 
fig = model.plot(forecast_data, uncertainty=True)

'''Even though we have a lot of places where the rate can possibly change, because of the sparse prior, most of these changepoints go unused. We can see this by plotting the magnitude of the rate change at each changepoint:'''
for cp in model.changepoints:
    plt.axvline(cp, c='gray', ls='--', lw=2)
    
deltas = model.params['delta'].mean(0)
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(range(len(deltas)), deltas, facecolor='#0072B2', edgecolor='#0072B2')
ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.4)
ax.set_ylabel('Rate change')
ax.set_xlabel('Potential changepoint')
fig.tight_layout()   

'''The number of potential changepoints can be set using the argument n_changepoints, but this is better tuned by adjusting the regularization. The locations of the signification changepoints can be visualized with:'''

from fbprophet.plot import add_changepoints_to_plot
fig = model.plot(forecast_data)
a = add_changepoints_to_plot(fig.gca(), model, forecast_data)

#####

forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

'''Plotting Prophet results
Prophet has a plotting mechanism called plot. This plot functionality draws the original data (black dots), the model (blue line) and the error of the forecast (shaded blue area).'''


model.plot(forecast_data);


'''Visualizing Prophet models
In order to build a useful dataframe to visualize our model versus our original data, we need to combine the output of the Prophet model with our original data set, then we'll build a new chart manually using pandas and matplotlib.

First, let's set our dataframes to have the same index of ds'''

df2.set_index('ds', inplace=True)
forecast_data.set_index('ds', inplace=True)

df3.head()
df2.head()
df3.set_index('DATE', inplace=True)
viz_df = df3.join(forecast_data[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')

'''If we look at the head(), we see the data has been joined correctly but the scales of our original data (sales) and our model (yhat) are different. We need to rescale the yhat colums(s) to get the same scale, so we'll use numpy's exp function to do that.
'''

viz_df.head()

viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])
viz_df.head()

'''Let's take a look at the sales and yhat_rescaled data together in a chart.'''

#Good
viz_df[['ASPUS_M', 'yhat_rescaled']].plot()



'''You can see from the chart that the model (blue) is pretty good when plotted against the actual signal (orange) but I like to make my vizualization's a little better to understand. To build my 'better' visualization, we'll need to go back to our original df3 and forecast dataframes.

First things first - we need to find the 2nd to last date of the original sales data in sales_df in order to ensure the original sales data and model data charts are connected.
'''

df3.index = pd.to_datetime(df3.index) #make sure our index as a datetime object
connect_date = df3.index[-2] #select the 2nd to last date

'''Using the connect_date we can now grab only the model data that after that date. To do this, we'll mask the forecast data.'''

mask = (forecast_data.index > connect_date)
predict_df = forecast_data.loc[mask]

predict_df.info()
predict_df.head()

'''Now, let's build a dataframe to use in our new visualization. We'll follow the same steps we did before.'''

viz_df = df3.join(predict_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
viz_df['yhat_scaled']=np.exp(viz_df['yhat'])

'''Now, if we take a look at the head() of viz_df we'll see 'NaN's everywhere except for our original data rows.
'''
viz_df.head()

'''
If we take a look at the tail() of the viz_df you'll see we have data for the forecasted data and NaN's for the original data series.'''

viz_df.tail()


'''time to plot
Now, let's plot everything to get the 'final' visualization of our sales data and forecast with errors.'''

fig, ax1 = plt.subplots()
ax1.plot(viz_df.ASPUS_M)
ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Housing Price (Orange) vs Hosing Growth Forecast (Black)')
ax1.set_ylabel('Housing Prices($)')
ax1.set_xlabel('Date')

L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual Housing Prices') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecasted Housing Prices') #change the legend text for 2nd plot



'''
This visualization is much better (in my opinion) than the default fbprophet plot. It is much easier to quickly understand and describe what's happening. The orange line is actual sales data and the black dotted line is the forecast. The gray shaded area is the uncertaintity estimation of the forecast.'''







































