import datetime
import pandas as pd
import gc
import numpy as np
from sklearn import preprocessing


def aug_ga_data_simple(train_df, test_df, const_cols):
    train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')

    train_df['date'] = train_df['date'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))


    test_df['date'] = test_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

    print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))

    # So apart from target variable, there is one more variable "trafficSource.campaignCode" not present in test dataset. So we need to remove this variable while building models. Also we can drop the constant variables which we got earlier.
    #
    # Also we can remove the "sessionId" as it is a unique identifier of the visit.

    # In[ ]:

    cols_to_drop = const_cols + ['sessionId']

    train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
    test_df = test_df.drop(cols_to_drop, axis=1)

    # Now let us create development and validation splits based on time to build the model. We can take the last two months as validation sample.

    # In[ ]:

    # Impute 0 for missing target values
    train_df["totals.transactionRevenue"].fillna(0, inplace=True)
    train_y = train_df["totals.transactionRevenue"].values
    train_id = train_df["fullVisitorId"].values
    test_id = test_df["fullVisitorId"].values

    # label encode the categorical variables and convert the numerical variables to float
    cat_cols = ["channelGrouping", "device.browser",
                "device.deviceCategory", "device.operatingSystem",
                "geoNetwork.city", "geoNetwork.continent",
                "geoNetwork.country", "geoNetwork.metro",
                "geoNetwork.networkDomain", "geoNetwork.region",
                "geoNetwork.subContinent", "trafficSource.adContent",
                "trafficSource.adwordsClickInfo.adNetworkType",
                "trafficSource.adwordsClickInfo.gclId",
                "trafficSource.adwordsClickInfo.page",
                "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
                "trafficSource.keyword", "trafficSource.medium",
                "trafficSource.referralPath", "trafficSource.source",
                'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
    for col in cat_cols:
        print(col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',
                'totals.newVisits']
    for col in num_cols:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)

    # Split the train dataset into development and valid based on time
    dev_df = train_df[train_df['date'] <= datetime.date(2017, 5, 31)]
    val_df = train_df[train_df['date'] > datetime.date(2017, 5, 31)]
    dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
    val_y = np.log1p(val_df["totals.transactionRevenue"].values)

    dev_X = dev_df[cat_cols + num_cols]
    val_X = val_df[cat_cols + num_cols]
    test_X = test_df[cat_cols + num_cols]

    return dev_X,dev_y, val_X,val_y, test_X


def explore_data_simple(train_df):
    train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
    gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('TransactionRevenue', fontsize=12)
    plt.show()

    # Wow, This confirms the first two lines of the competition overview.
    #     * The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.
    # Infact in this case, the ratio is even less.

    # In[ ]:

    nzi = pd.notnull(train_df["totals.transactionRevenue"]).sum()
    nzr = (gdf["totals.transactionRevenue"] > 0).sum()
    print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])
    print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])

    # So the ratio of revenue generating customers to customers with no revenue is in the ratio os 1.3%
    #
    # Since most of the rows have non-zero revenues, in the following plots let us have a look at the count of each category of the variable along with the number of instances where the revenue is not zero.
    #
    # ** Number of visitors and common visitors:**
    #
    # Now let us look at the number of unique visitors in the train and test set and also the number of common visitors.

    # In[ ]:

    print("Number of unique visitors in train set : ", train_df.fullVisitorId.nunique(), " out of rows : ",
          train_df.shape[0])
    print("Number of unique visitors in test set : ", test_df.fullVisitorId.nunique(), " out of rows : ",
          test_df.shape[0])
    print("Number of common visitors in train and test set : ",
          len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique()))))

    # **Columns with constant values: **
    #
    # Looks like there are quite a few features with constant value in the train set. Let us get the list of these features. As pointed by Svitlana in the comments below, let us not include the columns which has constant value and some null values.

    # In[ ]:

    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]
    const_cols

    # They are quite a few. Since the values are constant, we can just drop them from our feature list and save some memory and time in our modeling process.
    #
    # **Device Information:**

    # In[ ]:

    def horizontal_bar_chart(cnt_srs, color):
        trace = go.Bar(
            y=cnt_srs.index[::-1],
            x=cnt_srs.values[::-1],
            showlegend=False,
            orientation='h',
            marker=dict(
                color=color,
            ),
        )
        return trace

    # Device Browser
    cnt_srs = train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
    cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
    cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
    trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
    trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
    trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

    # Device Category
    cnt_srs = train_df.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
    cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
    cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
    trace4 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.8)')
    trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(71, 58, 131, 0.8)')
    trace6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.8)')

    # Operating system
    cnt_srs = train_df.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
    cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
    cnt_srs = cnt_srs.sort_values(by="count", ascending=False)


def make_xy_simple_ga(x,y,xtest, yfit_lnr, ypred_lnr):
    fam = 'BOOKS'
    nbr = '1'
    # plt.rcParams['figure.figsize'] = (15, 9)
    # plt.figure()
    # y.loc(axis=1)['sales', nbr, fam].plot()
    # yfit_lnr.loc(axis=1)['sales', nbr, fam].plot(label='Linear Regression')
    # # yfit_svr.loc(axis = 1)['sales', nbr, fam].plot(label = 'SVR')
    # # yfit_mean.loc(axis = 1)['sales', nbr, fam].plot(label = 'Mean')
    # # y.mean(axis = 1).plot()
    # # yfit_lnr.median(axis = 1).plot(label = 'Linear Regression')
    # # yfit_svr.median(axis = 1).plot(label = 'SVR')
    # # yfit_mean.mean(axis = 1).plot(label = 'Mean')
    # plt.legend()
    # plt.show()

    # You can concat linear regression's prediction with the training data, this is called blending.

    # In[26]:

    ymean = yfit_lnr.append(ypred_lnr)
    school = ymean.loc(axis=1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES']
    ymean = ymean.join(school.shift(1), rsuffix='lag1')  # I'm also adding school lag for it's cyclic yearly.
    x = x.loc['2017-05-01':]

    # In[27]:

    # ymean.loc['2017-08-16':]

    # In[28]:

    x = x.join(ymean)  # Concating linear result
    xtest = xtest.join(ymean)
    # display(x, xtest)

    # In[29]:

    y = y.loc['2017-05-01':]
    y

    # In[30]:

    print(y.isna().sum().sum())

    # In[31]:

    # display(x, xtest)
    return x,y

def make_x_ga(calendar, train, sdate,edate):
    y = train.unstack(['store_nbr', 'family']).loc[sdate:edate]
    fourier = CalendarFourier(freq='W', order=4)
    dp = DeterministicProcess(index=y.index,
                              order=1,
                              seasonal=False,
                              constant=False,
                              additional_terms=[fourier],
                              drop=True)
    x = dp.in_sample()
    x = x.join(calendar)
    return x

def prepare_test( calendar):
    xtest = dp.out_of_sample(steps=16)  # 16 because we are predicting next 16 days
    xtest = xtest.join(calendar)
    return xtest

def aug_calendar3(calendar):
    pass

    school_season = []  # Feature for school fluctuations
    for i, r in calendar.iterrows():
        if i.month in [4, 5, 8, 9]:
            school_season.append(1)
        else:
            school_season.append(0)
    calendar['school_season'] = school_season
    return calendar

def aug_calendar2(calendar, hol):
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
    return calendar

def lag_calendar(calendar):
    n_lags = 3
    for l in range(1, n_lags + 1):
        calendar[f'oil_lags{l}'] = calendar.avg_oil.shift(l)
    calendar.dropna(inplace=True)
    return calendar


def aug_calendar1(oil):
    pass

    calendar = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31')).to_period('D')

    calendar = calendar.join(oil.avg_oil)
    calendar['avg_oil'].fillna(method='ffill', inplace=True)
    calendar.dropna(inplace=True)
    return calendar

def populate_data(df=None):
    # Add data features
    df['date'] = pd.to_datetime(df['visitStartTime'])
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_dom'] = df['date'].dt.day
    df['sess_date_hour'] = df['date'].dt.hour
    # df['sess_date_week'] = df['date'].dt.weekofyear

    for f in ['transactionRevenue', 'visits', 'hits', 'pageviews', 'bounces', 'newVisits']:
        df[f] = multi_apply_func_on_series(
            df=df['totals'],
            func=get_dict_content,
            key=f,
            n_jobs=4
        )
        logger.info('Done with totals.{}'.format(f))

    for f in ['continent', 'subContinent', 'country', 'region', 'metro', 'city', 'networkDomain']:
        df[f] = multi_apply_func_on_series(
            df=df['geoNetwork'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with geoNetwork.{}'.format(f))

    for f in ['browser', 'operatingSystem', 'isMobile', 'deviceCategory']:
        df[f] = multi_apply_func_on_series(
            df=df['device'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with device.{}'.format(f))

    for f in ['source', 'medium']:
        df[f] = multi_apply_func_on_series(
            df=df['trafficSource'],
            func=get_dict_content_str,
            key=f,
            n_jobs=4
        )
        logger.info('Done with trafficSource.{}'.format(f))

    df.drop(['totals', 'geoNetwork', 'device', 'trafficSource', 'visitStartTime'], axis=1, inplace=True)

    # This is all Scirpus' fault :) https://www.kaggle.com/scirpus
    #
    df['dummy'] = 1
    df['user_cumcnt_per_day'] = (
                df[['fullVisitorId', 'date', 'dummy']].groupby(['fullVisitorId', 'date'])['dummy'].cumcount() + 1)
    df['user_sum_per_day'] = df[['fullVisitorId', 'date', 'dummy']].groupby(['fullVisitorId', 'date'])[
        'dummy'].transform(sum)
    df['user_cumcnt_sum_ratio_per_day'] = df['user_cumcnt_per_day'] / df['user_sum_per_day']
    df.drop('dummy', axis=1, inplace=True)

    return df


def factorize_categoricals(df=None, cat_indexers=None):
    # Find categorical features
    cat_feats = [f for f in df.columns
                 if ((df[f].dtype == 'object')
                     & (f not in ['fullVisitorId', 'sessionId', 'date',
                                  'totals', 'device', 'geoNetwork', 'device', 'trafficSource']))]
    logger.info('Categorical features : {}'.format(cat_feats))

    if cat_indexers is None:
        cat_indexers = {}
        for f in cat_feats:
            df[f], indexer = pd.factorize(df[f])
            cat_indexers[f] = indexer
    else:
        for f in cat_feats:
            df[f] = cat_indexers[f].get_indexer(df[f])

    return df, cat_indexers, cat_feats


def aggregate_sessions(df=None, cat_feats=None, sum_of_logs=False):
    """
    Aggregate session data for each fullVisitorId
    :param df: DataFrame to aggregate on
    :param cat_feats: List of Categorical features
    :param sum_of_logs: if set to True, revenues are first log transformed and then summed up
    :return: aggregated fullVisitorId data over Sessions
    """
    if sum_of_logs is True:
        # Go to log first
        df['transactionRevenue'] = np.log1p(df['transactionRevenue'])

    aggs = {
        'date': ['min', 'max'],
        'transactionRevenue': ['sum', 'size'],
        'hits': ['sum', 'min', 'max', 'mean', 'median'],
        'pageviews': ['sum', 'min', 'max', 'mean', 'median'],
        'bounces': ['sum', 'mean', 'median'],
        'newVisits': ['sum', 'mean', 'median']
    }

    for f in cat_feats + ['sess_date_dow', 'sess_date_dom', 'sess_date_hour']:
        aggs[f] = ['min', 'max', 'mean', 'median', 'var', 'std']

    users = df.groupby('fullVisitorId').agg(aggs)
    logger.info('User aggregation done')

    # This may not work in python 3.5, since keys ordered is not guaranteed
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    logger.info('New columns are : {}'.format(new_columns))
    users.columns = new_columns

    # Add dates
    users['date_diff'] = (users.date_max - users.date_min).astype(np.int64) // (24 * 3600 * 1e9)

    # Go to log if not already done
    if sum_of_logs is False:
        # Go to log first
        users['transactionRevenue_sum'] = np.log1p(users['transactionRevenue_sum'])

    return users


def aug_first_kernel(df_train, df_test, df_hist_trans, df_new_merchant_trans):
    for df in [df_hist_trans, df_new_merchant_trans]:
        df['category_2'].fillna(1.0, inplace=True)
        df['category_3'].fillna('A', inplace=True)
        df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

    # In[ ]:

    def get_new_columns(name, aggs):
        return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

    # In[ ]:

    for df in [df_hist_trans, df_new_merchant_trans]:
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df['year'] = df['purchase_date'].dt.year
        df['weekofyear'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df['dayofweek'] = df['purchase_date'].dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
        df['hour'] = df['purchase_date'].dt.hour
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
        df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
        df['month_diff'] += df['month_lag']

    # In[ ]:

    # display(df_hist_trans.head())

    # In[ ]:

    aggs = {}
    for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
                'merchant_category_id']:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2', 'category_3']:
        df_hist_trans[col + '_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns('hist', aggs)
    df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False, inplace=True)
    df_hist_trans_group['hist_purchase_date_diff'] = (
                df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
    df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff'] / \
                                                        df_hist_trans_group['hist_card_id_size']
    df_hist_trans_group['hist_purchase_date_uptonow'] = (
                datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
    df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
    del df_hist_trans_group;
    gc.collect()

    # In[ ]:

    aggs = {}
    for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_id',
                'merchant_category_id']:
        aggs[col] = ['nunique']
    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var']
    aggs['month_diff'] = ['mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2', 'category_3']:
        df_new_merchant_trans[col + '_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns('new_hist', aggs)
    df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False, inplace=True)
    df_hist_trans_group['new_hist_purchase_date_diff'] = (
                df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group[
            'new_hist_purchase_date_min']).dt.days
    df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff'] / \
                                                            df_hist_trans_group['new_hist_card_id_size']
    df_hist_trans_group['new_hist_purchase_date_uptonow'] = (
                datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
    df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
    del df_hist_trans_group;
    gc.collect()

    # In[ ]:

    del df_hist_trans;
    gc.collect()
    del df_new_merchant_trans;
    gc.collect()
    df_train.head(5)

    return df_train, df_test


def remove_outlier(df_train, df_test):
    pass
    df_train['outliers'] = 0
    df_train.loc[df_train['target'] < -30, 'outliers'] = 1
    df_train['outliers'].value_counts()


    # In[ ]:


    for df in [df_train,df_test]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['dayofweek'] = df['first_active_month'].dt.dayofweek
        df['weekofyear'] = df['first_active_month'].dt.weekofyear
        df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
        df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
        df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
        for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                         'new_hist_purchase_date_min']:
            df[f] = df[f].astype(np.int64) * 1e-9
        df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']
        df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

    for f in ['feature_1','feature_2','feature_3']:
        order_label = df_train.groupby([f])['outliers'].mean()
        df_train[f] = df_train[f].map(order_label)
        df_test[f] = df_test[f].map(order_label)

    return df_train, df_test

def remove_target(df_train):
    df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'target', 'outliers']]
    target = df_train['target']
    del df_train['target']
    return df_train,df_train_columns


def binarize_elo_world(new_transactions, historical_transactions):
    def binarize(df):
        for col in ['authorized_flag', 'category_1']:
            df[col] = df[col].map({'Y': 1, 'N': 0})
        return df

    historical_transactions = binarize(historical_transactions)
    new_transactions = binarize(new_transactions)
    return new_transactions, historical_transactions


def aug_elo_world():
    historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions[
        'purchase_date']).dt.days) // 30
    historical_transactions['month_diff'] += historical_transactions['month_lag']

    new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days) // 30
    new_transactions['month_diff'] += new_transactions['month_lag']

    # In[ ]:

    historical_transactions[:5]

    # In[ ]:

    historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
    new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])

    historical_transactions = reduce_mem_usage(historical_transactions)
    new_transactions = reduce_mem_usage(new_transactions)

    agg_fun = {'authorized_flag': ['mean']}
    auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
    auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
    auth_mean.reset_index(inplace=True)

    authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
    historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]

    # First, following [Robin Denz](https://www.kaggle.com/denzo123/a-closer-look-at-date-variables) analysis, I define a few dates features:

    # In[ ]:

    historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
    authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
    new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month

    # Then I define two functions that aggregate the info contained in these two tables. The first function aggregates the function by grouping on `card_id`:

    # In[ ]:

    def aggregate_transactions(history):
        history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']). \
                                              astype(np.int64) * 1e-9

        agg_func = {
            'category_1': ['sum', 'mean'],
            'category_2_1.0': ['mean'],
            'category_2_2.0': ['mean'],
            'category_2_3.0': ['mean'],
            'category_2_4.0': ['mean'],
            'category_2_5.0': ['mean'],
            'category_3_A': ['mean'],
            'category_3_B': ['mean'],
            'category_3_C': ['mean'],
            'merchant_id': ['nunique'],
            'merchant_category_id': ['nunique'],
            'state_id': ['nunique'],
            'city_id': ['nunique'],
            'subsector_id': ['nunique'],
            'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
            'installments': ['sum', 'mean', 'max', 'min', 'std'],
            'purchase_month': ['mean', 'max', 'min', 'std'],
            'purchase_date': [np.ptp, 'min', 'max'],
            'month_lag': ['mean', 'max', 'min', 'std'],
            'month_diff': ['mean']
        }

        agg_history = history.groupby(['card_id']).agg(agg_func)
        agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
        agg_history.reset_index(inplace=True)

        df = (history.groupby('card_id')
              .size()
              .reset_index(name='transactions_count'))

        agg_history = pd.merge(df, agg_history, on='card_id', how='left')

        return agg_history

    # In[ ]:

    history = aggregate_transactions(historical_transactions)
    history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
    history[:5]

    # In[ ]:

    authorized = aggregate_transactions(authorized_transactions)
    authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
    authorized[:5]

    # In[ ]:

    new = aggregate_transactions(new_transactions)
    new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
    new[:5]

    # The second function first aggregates on the two variables `card_id` and `month_lag`. Then a second grouping is performed to aggregate over time:

    # In[ ]:

    def aggregate_per_month(history):
        grouped = history.groupby(['card_id', 'month_lag'])

        agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
        }

        intermediate_group = grouped.agg(agg_func)
        intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
        intermediate_group.reset_index(inplace=True)

        final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
        final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
        final_group.reset_index(inplace=True)

        return final_group

    # ___________________________________________________________
    final_group = aggregate_per_month(authorized_transactions)
    final_group[:10]

    # In[ ]:

    def successive_aggregates(df, field1, field2):
        t = df.groupby(['card_id', field1])[field2].mean()
        u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
        u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
        u.reset_index(inplace=True)
        return u

    # In[ ]:

    additional_fields = successive_aggregates(new_transactions, 'category_1', 'purchase_amount')
    additional_fields = additional_fields.merge(
        successive_aggregates(new_transactions, 'installments', 'purchase_amount'),
        on='card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(new_transactions, 'city_id', 'purchase_amount'),
                                                on='card_id', how='left')
    additional_fields = additional_fields.merge(successive_aggregates(new_transactions, 'category_1', 'installments'),
                                                on='card_id', how='left')
    return history, authorized, new, final_group, auth_mean, additional_fields


def join_fields_elo_world(train, test, history, authorized, new, final_group, auth_mean, additional_fields):
    pass
    train = pd.merge(train, history, on='card_id', how='left')
    test = pd.merge(test, history, on='card_id', how='left')

    train = pd.merge(train, authorized, on='card_id', how='left')
    test = pd.merge(test, authorized, on='card_id', how='left')

    train = pd.merge(train, new, on='card_id', how='left')
    test = pd.merge(test, new, on='card_id', how='left')

    train = pd.merge(train, final_group, on='card_id', how='left')
    test = pd.merge(test, final_group, on='card_id', how='left')

    train = pd.merge(train, auth_mean, on='card_id', how='left')
    test = pd.merge(test, auth_mean, on='card_id', how='left')

    train = pd.merge(train, additional_fields, on='card_id', how='left')
    test = pd.merge(test, additional_fields, on='card_id', how='left')

    return train, test


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(train, test, start_time=time()):
    train = train[train.price > 0.0].reset_index(drop=True)
    print('Train shape without zero price: ', train.shape)

    nrow_train = train.shape[0]
    y_train = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])

    del train
    del test
    gc.collect()

    merge['has_category'] = (merge['category_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_category filled.')

    merge['category_name'] = merge['category_name'] \
        .fillna('other/other/other') \
        .str.lower() \
        .astype(str)
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'], merge['gen_subcat1'] = \
        zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    print(f'[{time() - start_time}] Split categories completed.')

    merge['has_brand'] = (merge['brand_name'].notnull()).astype('category')
    print(f'[{time() - start_time}] Has_brand filled.')

    merge['gencat_cond'] = merge['general_cat'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_1_cond'] = merge['subcat_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
    merge['subcat_2_cond'] = merge['subcat_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
    print(f'[{time() - start_time}] Categories and item_condition_id concancenated.')

    merge['name'] = merge['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['brand_name'] = merge['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    merge['item_description'] = merge['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace(to_replace='No description yet', value='')
    print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(merge, start_time)

    brands_filling(merge)
    print(f'[{time() - start_time}] Brand name filled.')

    merge['name'] = merge['name'] + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Name concancenated.')

    merge['item_description'] = merge['item_description'] \
                                + ' ' + merge['name'] \
                                + ' ' + merge['subcat_1'] \
                                + ' ' + merge['subcat_2'] \
                                + ' ' + merge['general_cat'] \
                                + ' ' + merge['brand_name']
    print(f'[{time() - start_time}] Item description concatenated.')

    merge.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)

    return merge, y_train, nrow_train


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res


