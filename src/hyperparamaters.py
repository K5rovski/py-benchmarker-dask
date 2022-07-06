#!/usr/bin/env python
# coding: utf-8

# ## Please upvote before fork!!!
# This notebook is a hyperparameter-ed version from [BIZEN's](https://www.kaggle.com/hiro5299834) [Notebook](https://www.kaggle.com/hiro5299834/store-sales-ridge-voting-bagging-et-bagging-rf)
# Please upvote that notebook too if you find it useful :)

# # Import Library

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, style
style.use('seaborn-darkgrid')
import seaborn as sns
sns.set_style('darkgrid')
from plotly import express as px, graph_objects as go

from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor

import gc
gc.enable()
from warnings import filterwarnings, simplefilter
filterwarnings('ignore')
simplefilter('ignore')


# In[2]:


rcParams['figure.figsize'] = (12, 9) # Konfigurasi jendela figure


# # Fetching dataset

# In[3]:


train = pd.read_csv('../input/store-sales-time-series-forecasting/train.csv',
                    parse_dates = ['date'], infer_datetime_format = True,
                    dtype = {'store_nbr' : 'category',
                             'family' : 'category'},
                    usecols = ['date', 'store_nbr', 'family', 'sales'])
train['date'] = train.date.dt.to_period('D')
train = train.set_index(['date', 'store_nbr', 'family']).sort_index()
train


# In[4]:


test = pd.read_csv('../input/store-sales-time-series-forecasting/test.csv',
                   parse_dates = ['date'], infer_datetime_format = True)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['date', 'store_nbr', 'family']).sort_values('id')
test


# # Calendar Engineering

# In[5]:


oil = pd.read_csv('../input/store-sales-time-series-forecasting/oil.csv',
                  parse_dates=['date'], infer_datetime_format=True,
                  index_col='date').to_period('D')
oil['avg_oil'] = oil['dcoilwtico'].rolling(7).mean()


calendar = pd.DataFrame(index = pd.date_range('2013-01-01', '2017-08-31')).to_period('D')
oil = pd.read_csv('../input/store-sales-time-series-forecasting/oil.csv',
                  parse_dates = ['date'], infer_datetime_format = True,
                  index_col = 'date').to_period('D')
oil['avg_oil'] = oil['dcoilwtico'].rolling(7).mean()
calendar = calendar.join(oil.avg_oil)
calendar['avg_oil'].fillna(method = 'ffill', inplace = True)
calendar.dropna(inplace = True)


# We make date in calendar from beginning of train until last date of test.
# 
# We also concatenate calendar with oil price.

# In[6]:


# Plotting oil price
_ = sns.lineplot(data = oil.dcoilwtico.to_timestamp())


# You can see that oil price is only high at 2013 to 2014, however in 2015 it's starting to go down.
# 
# So, because we only predict 16 data points we will only need the training data from at least 2015

# In[7]:


_ = plot_pacf(calendar.avg_oil, lags = 12) # Lagplot oil price (Feature Engineering)


# You can see that max value for making a lags is up to 5, but you can take whatever you want.
# 
# I'm taking 3 lags of oil

# # Adding lags

# In[8]:


n_lags = 3
for l in range(1, n_lags + 1) :
    calendar[f'oil_lags{l}'] = calendar.avg_oil.shift(l)
calendar.dropna(inplace = True)
calendar


# # Correlation plot

# In[9]:


lag = 'oil_lags1'
plt.figure()
sns.regplot(x = calendar[lag], y = calendar.avg_oil)
plt.title(f'corr {calendar.avg_oil.corr(calendar[lag])}')
plt.show()


# # Fetching holiday dataset

# In[10]:


hol = pd.read_csv('../input/store-sales-time-series-forecasting/holidays_events.csv',
                  parse_dates = ['date'], infer_datetime_format = True,
                  index_col = 'date').to_period('D')
hol = hol[hol.locale == 'National'] # I'm only taking National holiday so there's no false positive.
hol = hol.groupby(hol.index).first() # Removing duplicated holiday at the same date
hol


# # Feature Engineering for holiday

# In[11]:


calendar = calendar.join(hol) # Joining calendar with holiday dataset
calendar['dofw'] = calendar.index.dayofweek # Weekly day
calendar['wd'] = 1
calendar.loc[calendar.dofw > 4, 'wd'] = 0 # If it's saturday or sunday then it's not Weekday
calendar.loc[calendar.type == 'Work Day', 'wd'] = 1 # If it's Work Day event then it's a workday
calendar.loc[calendar.type == 'Transfer', 'wd'] = 0 # If it's Transfer event then it's not a work day
calendar.loc[calendar.type == 'Bridge', 'wd'] = 0 # If it's Bridge event then it's not a work day
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == False), 'wd'] = 0 # If it's holiday and the holiday is not transferred then it's holiday
calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == True), 'wd'] = 1 # If it's holiday and transferred then it's not holiday
calendar = pd.get_dummies(calendar, columns = ['dofw'], drop_first = True) # One-hot encoding (Make sure to drop one of the columns by 'drop_first = True')
calendar = pd.get_dummies(calendar, columns = ['type']) # One-hot encoding for type holiday (No need to drop one of the columns because there's a "No holiday" already)
calendar.drop(['locale', 'locale_name', 'description', 'transferred'], axis = 1, inplace = True) # Unused columns
calendar


# In[12]:


#calendar['wd_lag1'] = calendar.wd.shift(1)
#calendar['wd_fore1'] = calendar.wd.shift(-1).fillna(0)
#calendar.dropna(inplace = True)
#calendar


# # Dependent Variable Viz

# In[13]:


y = train.unstack(['store_nbr', 'family']).loc['2016-06':'2017']
family = {c[2] for c in train.index}
for f in family :
    ax = y.loc(axis = 1)['sales', :, f].plot(legend = None)
    ax.set_title(f)


# Graphs above are the visualization of each product

# In[14]:


sdate = '2017-04-30' # Start and end of training date
edate = '2017-08-15'


# In[15]:


school_season = [] # Feature for school fluctuations
for i, r in calendar.iterrows() :
    if i.month in [4, 5, 8, 9] :
        school_season.append(1)
    else :
        school_season.append(0)
calendar['school_season'] = school_season
calendar


# # DeterministicProcess

# In[16]:


y = train.unstack(['store_nbr', 'family']).loc[sdate:edate]
fourier = CalendarFourier(freq = 'W', order = 4)
dp = DeterministicProcess(index = y.index,
                          order = 1,
                          seasonal = False,
                          constant = False,
                          additional_terms = [fourier],
                          drop = True)
x = dp.in_sample()
x = x.join(calendar)
x


# In[17]:


print(y.isna().sum().sum())
display(y)


# In[18]:


xtest = dp.out_of_sample(steps = 16) # 16 because we are predicting next 16 days
xtest = xtest.join(calendar)
xtest


# In[19]:


def make_lags(x, lags = 1) : #Fungsi untuk membuat fitur lags
    lags = lags
    x_ = x.copy()
    for i in range(lags) :
        lag = x_.shift(i + 1)
        x = pd.concat([x, lag], axis = 1)
    return x


# # Using LinearRegression to make a generalized line (It's usually called blending.)

# In[20]:


from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

# As you can see, RMSLE will have higher "tolerance" for higher value meanwhile Mean Absolute Error will go as usual.
# 
# Therefore, I will take the Linear Regression's result instead because it has lower MAE and MAE is reliable because it's robust to outlier

# I'm not gonna use validation data because the data we have is not much and because we are using linear-based algorithm so only using training would be fine.

# In[23]:


display(x, xtest)


# In[24]:


ypred_svr


# In[25]:


fam = 'BOOKS'
nbr = '1'
plt.rcParams['figure.figsize'] = (15, 9)
plt.figure()
y.loc(axis = 1)['sales', nbr, fam].plot()
yfit_lnr.loc(axis = 1)['sales', nbr, fam].plot(label = 'Linear Regression')
#yfit_svr.loc(axis = 1)['sales', nbr, fam].plot(label = 'SVR')
#yfit_mean.loc(axis = 1)['sales', nbr, fam].plot(label = 'Mean')
#y.mean(axis = 1).plot()
#yfit_lnr.median(axis = 1).plot(label = 'Linear Regression')
#yfit_svr.median(axis = 1).plot(label = 'SVR')
#yfit_mean.mean(axis = 1).plot(label = 'Mean')
plt.legend()
plt.show()


# You can concat linear regression's prediction with the training data, this is called blending.

# In[26]:


ymean = yfit_lnr.append(ypred_lnr)
school = ymean.loc(axis = 1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES']
ymean = ymean.join(school.shift(1), rsuffix = 'lag1') # I'm also adding school lag for it's cyclic yearly.
x = x.loc['2017-05-01':]


# In[27]:


ymean.loc['2017-08-16':]


# In[28]:


x = x.join(ymean) # Concating linear result
xtest = xtest.join(ymean)
display(x, xtest)


# In[29]:


y = y.loc['2017-05-01':]
y


# In[30]:


print(y.isna().sum().sum())


# In[31]:


display(x, xtest)


# This is the model I use, as I said I'm taking it from [BIZEN](https://www.kaggle.com/hiro5299834) and modifying it.

# # Model Creation

# In[32]:


from joblib import Parallel, delayed
import warnings

# Import necessary library
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor

# SEED for reproducible result
SEED = 5

class CustomRegressor():
    
    def __init__(self, n_jobs=-1, verbose=0):
        
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.estimators_ = None
        
    def _estimator_(self, X, y):
    
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        if y.name[2] == 'SCHOOL AND OFFICE SUPPLIES': # Because SCHOOL AND OFFICE SUPPLIES has weird trend, we use decision tree instead.
            r1 = ExtraTreesRegressor(n_estimators = 225, n_jobs=-1, random_state=SEED)
            r2 = RandomForestRegressor(n_estimators = 225, n_jobs=-1, random_state=SEED)
            b1 = BaggingRegressor(base_estimator=r1,
                                  n_estimators=10,
                                  n_jobs=-1,
                                  random_state=SEED)
            b2 = BaggingRegressor(base_estimator=r2,
                                  n_estimators=10,
                                  n_jobs=-1,
                                  random_state=SEED)
            model = VotingRegressor([('et', b1), ('rf', b2)]) # Averaging the result
        else:
            ridge = Ridge(fit_intercept=True, solver='auto', alpha=0.75, normalize=True, random_state=SEED)
            svr = SVR(C = 0.2, kernel = 'rbf')
            
            model = VotingRegressor([('ridge', ridge), ('svr', svr)]) # Averaging result
        model.fit(X, y)

        return model

    def fit(self, X, y):
        from tqdm.auto import tqdm
        
        
        if self.verbose == 0 :
            self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                  verbose=0,
                                  )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in range(y.shape[1]))
        else :
            print('Fit Progress')
            self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                  verbose=0,
                                  )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in tqdm(range(y.shape[1])))
        return
    
    def predict(self, X):
        from tqdm.auto import tqdm
        if self.verbose == 0 :
            y_pred = Parallel(n_jobs=self.n_jobs, 
                              verbose=0)(delayed(e.predict)(X) for e in self.estimators_)
        else :
            print('Predict Progress')
            y_pred = Parallel(n_jobs=self.n_jobs, 
                              verbose=0)(delayed(e.predict)(X) for e in tqdm(self.estimators_))
        
        return np.stack(y_pred, axis=1)


# In[33]:


get_ipython().run_cell_magic('time', '', '\nmodel = CustomRegressor(n_jobs=-1, verbose=1)\nmodel.fit(x, y)\ny_pred = pd.DataFrame(model.predict(x), index=x.index, columns=y.columns)\n')


# In[34]:


display(y_pred)
print(y_pred.isna().sum().sum())


# # Evaluation

# In[35]:


from sklearn.metrics import mean_squared_log_error
y_pred = y_pred.stack(['store_nbr', 'family']).clip(0.)
y_ = y.stack(['store_nbr', 'family']).clip(0.)

y_['pred'] = y_pred.values
print(y_.groupby('family').apply(lambda r : np.sqrt(np.sqrt(mean_squared_log_error(r['sales'], r['pred'])))))
# Looking at error
print('RMSLE : ', np.sqrt(np.sqrt(msle(y_['sales'], y_['pred']))))


# All seems good.

# In[36]:


y_pred.isna().sum()


# In[37]:


ypred = pd.DataFrame(model.predict(xtest), index = xtest.index, columns = y.columns).clip(0.)
ypred


# In[38]:


ypred = ypred.stack(['store_nbr', 'family'])
ypred


# # Submission

# In[39]:


sub = pd.read_csv('../input/store-sales-time-series-forecasting/sample_submission.csv')
sub['sales'] = ypred.values
sub.to_csv('submission.csv', index = False) # Submit
sub


# Thank you!!!
