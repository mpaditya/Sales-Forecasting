# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:42:08 2019

@author: PrabhakaronA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import statsmodels.imputation.mice as mice
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


df_channels = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning//Hackathons//Ticket Sales//ForecastCase//orders_channels.csv")
df_country = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning//Hackathons//Ticket Sales//ForecastCase//orders_country.csv")
df_tickets = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning//Hackathons//Ticket Sales//ForecastCase//orders_tickets.csv")


""" Checking no of missing values in each file   """
#print("missing values in Channels \n ", df_channels.apply(lambda x: sum(x.isnull()),axis=0))
#print("missing values in Country \n ", df_country.apply(lambda x: sum(x.isnull()),axis=0))
#print("missing values in Tickets \n ", df_tickets.apply(lambda x: sum(x.isnull()),axis=0))


""" No missing values in tickets file. No missing values in ID column in any file. Channel_id & Country_2
    have high no of missing values     """          


"""   Channels and Tickets have 1970544 unique IDs but Country file has only 1917668 unique IDs. So, 52876 IDs
      are missing in Country file. Also, since size of Channels file is higher than Tickets & Country,
      it has multiple entries for some IDs which the other 2 files don't have. """
      
ch_tick = pd.merge(df_channels,df_tickets, how='outer', on = ['id'])
#print("missing values in Outer Join of channels tickets \n ", ch_tick.apply(lambda x: sum(x.isnull()),axis=0), '\n')

df = pd.merge(ch_tick,df_country, how='outer', on = ['id'])

print("missing values in Outer Join of All Files \n ", df.apply(lambda x: sum(x.isnull()),axis=0), '\n')

df = df.sort_values(by=['date'])


#df.describe()
#df['type'].value_counts()
#df['channel_id'].value_counts()


df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)
df['day_of_week'] = df['date'].apply(lambda x: x.weekday())



"""      *************   LABEL ENCODING OF TYPE  *******************       """
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['type'] = le.fit_transform(df['type'])


#   ********** IMPUTE MISSING VALUE IN COUNTRY_1 WITH COUNTRY_2    ************   

df.loc[(df['country_2'] == 'xx'), 'country_2'] = np.nan
# Substitute country_1 NaNs with country_2
df.loc[df['country_1'].isna(), 'country_1'] = df.loc[df['country_1'].isna(), 'country_2']
# No of missing values in country_1 reduces from 74718 to  63711

#   ********** CONVERT COUNTRY_1 TO FLOAT    ************   

df['country_1'] = df['country_1'].astype(str).astype(float)
df['country_2'] = df['country_2'].astype(str).astype(float)


#   ********** DROP ALL ROWS WITH NaN    ************   
columns = ['n_tickets', 'type', 'country_1', 'channel_id']
df = df.dropna(subset = columns)


"""  **************  SELECTING TOP CATEGORIES OF CHANNEL AND COUNTRY  *******************  """

df.loc[(pd.isnull(df['channel_id'])), 'channel_id'] = df["channel_id"].mode()   

#df['channel_id'] = df['channel_id'].fillna(0)

ch_values = df['channel_id'].value_counts().reset_index()
ch_top5 = ch_values.iloc[0: 5, 0]
#ch_values.iloc[5: ,].sum()

df.loc[(~df['channel_id'].isin(ch_top5.values)), 'channel_id'] = df.loc[(~df['channel_id'].isin(ch_top5)), 'channel_id'].apply(lambda x: 100.0)

coun_values = df['country_1'].value_counts().reset_index()
coun_top5 = coun_values.iloc[0: 5, 0]
coun_values.iloc[5: ,].sum()

df.loc[(~df['country_1'].isin(coun_top5)), 'country_1'] = df.loc[(~df['country_1'].isin(coun_top5.values)), 'country_1'].apply(lambda x: 100.0)


#   ********** CONVERT CHANNEL_ID & COUNTRY_1 TO int    ************ 

df['channel_id'] = df['channel_id'].astype(int)
df['country_1'] = df['country_1'].astype(int)

"""   We need to split the data into train & test before doing any missing value imputation because the
test data information must not be leaked and used to impute values in the training set   """


"""   Since we have only 1 full year's data, we will use all 2017 data for training and set aside 2018 data
for testing/validation   """

df_train = df.loc[(df['year'] == 2017)]
df_test = df.loc[(df['year'] == 2018)]


#   ********** DROP UNWANTED COLUMNS    ************   
df = df.drop(['id', 'country_2', 'day', 'year'], axis=1)
df_train = df_train.drop(['id', 'country_2', 'day', 'year'], axis=1)
df_test = df_test.drop(['id', 'country_2', 'day', 'year'], axis=1)

"""      *************   EDA       *******************       """

#sns.countplot(x="country_1", hue="type",  data=df)
#fig=plt.gcf()
#fig.set_size_inches(10,10)
#plt.show()

#sns.violinplot(data = df, x = "type", y = "n_tickets")
#sns.boxplot(data = df, x = "channel_id", y = "n_tickets")
#sns.boxplot(data = df, x = "country_1", y = "n_tickets")
#fig=plt.gcf()
#fig.set_size_inches(10,10)
#plt.show()

# can also be visualized using histograms for all the continuous variables.

#sns.distplot(df["n_tickets"], bins = 20, kde = False, color ='#ff4125')
#plt.title("Distribution of n_tickets")
#fig=plt.gcf()
#fig.set_size_inches(10,10)
#plt.show()
         
"""             **********       Correlation Maps     ********           """

#corr = df.drop(['date'], axis=1).dropna().corr(method='pearson')

#f,ax = plt.subplots(figsize=(10, 10))
#sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#from scipy.stats import spearmanr
#rho, pvalue = spearmanr(df.drop(['date', 'id', 'year', 'day'], axis=1).dropna())
#f,ax = plt.subplots(figsize=(10, 10))
#sns.heatmap(rho, annot=True, linewidths=.5, fmt= '.1f',ax=ax)


"""      *************   MISSING VALUE HANDLING       *******************       """         


#       ******   IMPUTATION FOR CHANNEL  **********


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def rf_impute (df, columns):
    

    df_with_channel = df[pd.isnull(df_train['channel_id']) == False]
    df_without_channel = df[pd.isnull(df_train['channel_id'])]
    
    df_with_channel = df_with_channel.dropna(subset = columns)
    df_without_channel = df_without_channel.dropna(subset = columns)

    rf_channel = SVR()
    rf_channel.fit(df_with_channel[columns], df_with_channel['channel_id'])
    pred_channel = rf_channel.predict(df_without_channel[columns])
    print(rf_channel.score(df_with_channel[columns], df_with_channel['channel_id']))
    
    pred_channel = np.rint(pred_channel)
    unique_elements, counts_elements = np.unique(pred_channel, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))

    df_without_channel['channel_id'] = pred_channel
    
    return pred_channel, df_without_channel

#pred_channel_train, df_train_without_channel  = rf_impute(df_train, columns)




#    ******       STATISTICS BASED IMPUTATION      **********

#df_train.loc[(pd.isnull(df_train['channel_id'])), 'channel_id'] = df_train["channel_id"].mode()
#df_test.loc[(pd.isnull(df_test['channel_id'])), 'channel_id'] = df_test["channel_id"].mode()    


temp2 = df.set_index('date')
#temp2 = temp2.resample("1D").sum()
#plt.plot(temp4['n_tickets'])



""" First we need to create a Dict which will stor every combination of (channel, country) as a key and the 
corresponding values as a dataframe. Then, we need to group by date to get one value for each day. We aggregate
this grouping by using sum to get the total no of tickets for every day for that combination of (channel, country).
However, type and month also gets summed. So we need to create another temporary variable and store the mode (most
frequently occuring value) for type and month. And we use the key to fill value of channel and country.  """

#temp3 = df.groupby(['date','channel_id', 'country_1'])
#temp4 = df.groupby(['date','channel_id', 'country_1']).sum()
#temp = temp3.agg(lambda x:x.value_counts().index[0])

def fbmodel(dff):  
    m = Prophet(growth = 'linear', yearly_seasonality = True, daily_seasonality = False, weekly_seasonality = True, seasonality_prior_scale= 0.1)
    #dff['cap'] = 100
    dff['floor'] = 0
    m.fit(dff)
    future = m.make_future_dataframe(periods = 57, freq = 'D', include_history = False)
    #future['cap'] = 1.2
    #future['floor'] = 0
    forecast = m.predict(future)
    return (forecast[['ds', 'yhat']])                                                                                                              
                                                                                                              
def create_ts(df, df_date):
    
    df_group = {}
    temp5 = df.groupby(['channel_id', 'country_1'])

    for name, group in temp5:
        df_group[name] = group
        df_group[name] = df_group[name].groupby(['date']).sum()
        temp = group.groupby(['date'])
        temp = temp.agg(lambda x:x.value_counts().index[0])
        df_group[name]['type'] = temp['type']
        df_group[name]['month'] = temp['month']
        df_group[name]['channel_id'] = name[0]
        df_group[name]['country_1'] = name[1]
    
        """  Now we need to create all days from 01 Jan 2017 to 26 Feb 2018 and merge it with every dataframe in 
        df_group.Then we will fill the missing values for channel, country. For type, we will fill it with mode. 
        For month,just extract the month from the date. For tickets, just fill 0 since no tickets were sold on those
        days for that particular combination of (channel, country)    """

        
        df_group[name] = df_group[name].merge(df_date, how = 'right', on = 'date')
        df_group[name].drop(['type'], axis = 1, inplace = True)
        df_group[name]['day_of_week'] = df_group[name]['date'].apply(lambda x: x.weekday())
        df_group[name]['n_tickets'].fillna(0, inplace = True)
        df_group[name]['channel_id'].fillna(name[0], inplace = True)
        df_group[name]['country_1'].fillna(name[1], inplace = True)
#        df_group[name].set_index('date', inplace = True)
#        df_group[name].sort_index(inplace = True)
    return df_group


#t_index = pd.DatetimeIndex(start='2017-01-01 00:00:00', end='2018-02-26 00:00:00', freq='1D')
#df_date = pd.DataFrame(t_index, columns = ['date'])

t_index_train = pd.DatetimeIndex(start='2017-01-01 00:00:00', end='2017-12-31 00:00:00', freq='1D')
df_date_train = pd.DataFrame(t_index_train, columns = ['date'])
df_group_train = create_ts(df_train, df_date_train)

t_index_test = pd.DatetimeIndex(start='2018-01-01 00:00:00', end='2018-02-26 00:00:00', freq='1D')
df_date_test = pd.DataFrame(t_index_test, columns = ['date'])
df_group_test = create_ts(df_test, df_date_test)



""" ************************************   PROPHET MODEL FOR EACH TS   ******************************     """

from fbprophet import Prophet
#series_fb['dust-combined'], lambda_prophet = stats.boxcox(series_fb['dust-combined'])
pred= {}
for key, value in df_group_train.items():
    df_group_train[key].reset_index(inplace = True)
    series_fb = df_group_train[key][["n_tickets", "date"]].rename(columns={"n_tickets": "y", "date": "ds"})
    pred[key] = fbmodel(series_fb)
    pred[key]['yhat'] = np.rint(pred[key]['yhat'])

mse = []
unique = []
from sklearn.metrics import mean_squared_error
for key in df_group_test:
    mse.append(mean_squared_error(df_group_test[key]['n_tickets'], pred[key]['yhat']))
    unique.append(pred[key]['yhat'].unique())
#    mse.index = key
    
#forecast = pred[(1.0, 4.0)]
#plt.plot(forecast.set_index('ds'))
#plt.plot(actual['n_tickets'])
#fig=plt.gcf()
#fig.set_size_inches(20,15)
#plt.show()

#plt.plot(df_group_train[key]['n_tickets'])
#plt.title(' Time Series of n_tickets for channel_id = 1 and country_1 = 4')
#plt.ylabel('Tickets sold')
#plt.xlabel('Date')


""" ********************************************* PROPHET MODEL DAILY SALES ********************************************* """

sales_tickets = df.groupby(['date']).sum()
sales_tickets.reset_index(inplace=True)
sales_tickets['year'] = sales_tickets['date'].apply(lambda x: x.year)
#sales_tickets_train = sales_tickets.iloc[ :-10, :]
#sales_tickets_test = sales_tickets.iloc[-10 :, :]
sales_tickets_train = sales_tickets[sales_tickets['year'] == 2017]
sales_tickets_test = sales_tickets[sales_tickets['year'] == 2018]

sales_tickets_train = sales_tickets_train[["n_tickets", "date"]].rename(columns={"n_tickets": "y", "date": "ds"})

m = Prophet(growth = 'linear', yearly_seasonality = True, daily_seasonality = False, weekly_seasonality = True, seasonality_prior_scale= 0.1)

#dff['cap'] = 100
#dff['floor'] = 0
m.fit(sales_tickets_train)

future = m.make_future_dataframe(periods = 57, freq = 'D', include_history = False)
#future['cap'] = 1.2
#future['floor'] = 0
forecast = m.predict(future)
#m.plot(forecast)
#m.plot_components(forecast)    

from sklearn.metrics import mean_squared_error
print('MSE of Prophet model for daily sales is: ', mean_squared_error(sales_tickets_test['n_tickets'], forecast['yhat']), '\n')

print('MSE of Prophet model for daily sales is: ', mean_squared_error(sales_tickets_test['n_tickets'], sales_tickets_rf), '\n')

from sklearn.metrics import r2_score
print('r2_score of Prophet model for daily sales is: ', r2_score(sales_tickets_test['n_tickets'], forecast['yhat']), '\n')

print('r2_score of RF model for daily sales is: ', r2_score(sales_tickets_test['n_tickets'], sales_tickets_rf), '\n')


plt.plot(forecast.set_index('ds')['yhat'])
plt.plot(sales_tickets_rf)
plt.plot(sales_tickets_test.set_index('date')['n_tickets'])
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.legend(['Prophet Forecast', 'RF Forecast', 'Actual Daily Sales'])
plt.title('Actual Vs Forecast of Daily Tickets Sales for Jan-Feb 2018')
plt.show()



#################    Check trend and seasonality of Daily Tickets Sales ########################

sales_tickets = df.groupby(['date']).sum()
plt.plot(sales_tickets['n_tickets'])
plt.title('Daily Sales of Tickets as Time Series')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()


channel_median = df[df['channel_id'] == 29.0]
channel_median = channel_median.groupby(['date']).sum() 
plt.plot(channel_median['n_tickets'])
plt.title('Daily Sales of Tickets through Channel ID 37')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()

"""   ************************** TIME SERIES DECOMPOSITION ******************************   """
from statsmodels.tsa.seasonal import seasonal_decompose

np.warnings.filterwarnings("ignore")
np.warnings.resetwarnings()
result = seasonal_decompose(sales_tickets['n_tickets'], freq = 7, model='additive')
result.plot()
fig=plt.gcf()
fig.set_size_inches(20,15)
plt.show()
residual = result.resid
residual.dropna(inplace = True)




""" ***************** FUNCTION FOR PLOTTING THE AUTOCORRELATION  ************************   """


def ACF_plot(timeseries, lags):
    
    np.warnings.filterwarnings("ignore")
    np.warnings.resetwarnings()
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(timeseries, lags=lags)
    plt.ylabel('Daily Ticket Sales Autocorrelation')
    plt.xlabel('Lags')
    plt.figure(figsize=(50,10))
    plt.show()

def PACF_plot(timeseries, lags):
    
    np.warnings.filterwarnings("ignore")
    np.warnings.resetwarnings()
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(timeseries, lags=lags)
    plt.ylabel('Daily Ticket Sales Partial Autocorrelation')
    plt.xlabel('Lags')
    plt.figure(figsize=(50,10))
    plt.show()
    
    
#ACF_plot(sales_tickets['n_tickets'], 30)
#PACF_plot(sales_tickets['n_tickets'], 30)

"""   ************************** ESTIMATING AND ELIMINATING SEASONALITY *****************************************   """

series = sales_tickets['n_tickets']
# Additive difference
series_diff = series.diff(7)


# CHECKING CORRELATION VALUES BETWEEN f(t) AND f(t-1)
from pandas.plotting import lag_plot
lag_plot(series.iloc[0:365], lag = 7)
plt.title('Plotting relationship of ticket sales with previous week (same day) sales')
plt.show()


shift = pd.concat([series, series.shift(1), series.shift(2), series.shift(3), series.shift(4), series.shift(5),
series.shift(6), series.shift(7)], axis = 1)
    
shift.columns = ['t', 't-1', 't-2', 't-3', 't-4', 't-5', 't-6', 't-7']
result = shift.corr()
#print(result)

sales_type_tickets = print(df.groupby(['type']).mean()['n_tickets'])



##############   Outlier Removal in Tickets
df['n_tickets'].apply(lambda x: np.abs(x - np.mean(df['n_tickets'])) / np.std(df['n_tickets']) > 4)





df_final = df_date_forecast[['date', 'country_1', 'channel_id', 'n_tickets']]
df_final.set_index('date', inplace = True)
df_final.to_csv("C://Users//prabhakarona//Documents//Digital General//Learning//Hackathons//Ticket Sales//ForecastCase//output.csv")


temp = df.groupby(['channel_id', 'country_1']).mean()
temp1 = df.groupby(['channel_id', 'country_1']).count()

temp2 = (temp['n_tickets']*temp1['n_tickets'])/422 

temp = df_final.groupby(['date']).sum()