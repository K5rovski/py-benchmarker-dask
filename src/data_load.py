import pandas as pd


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

def read_file(file_name=None, nrows=None):
    logger.info('Reading {}'.format(file_name))
    return pd.read_csv(
        file_name,
        usecols=['channelGrouping', 'date', 'fullVisitorId', 'sessionId', 'totals', 'device', 'geoNetwork', 'socialEngagementType', 'trafficSource', 'visitStartTime'],
        dtype={
            'channelGrouping': str,
            'geoNetwork': str,
            'date': str,
            'fullVisitorId': str,
            'sessionId': str,
            'totals': str,
            'device': str,
        },
        nrows=nrows,  # 50000
    )
def get_user_data(file_name='../input/train.csv', cat_indexers=None, nrows=None, sum_of_logs=False):
    data = read_file(file_name=file_name, nrows=nrows)
    logger.info('Data shape = {}'.format(data.shape))

    data = populate_data(df=data)

    data, cat_indexers, cat_feats = factorize_categoricals(df=data, cat_indexers=cat_indexers)

    users = aggregate_sessions(df=data, cat_feats=cat_feats, sum_of_logs=sum_of_logs)

    del data
    gc.collect()

    y = users['transactionRevenue_sum']
    users.drop(['date_min', 'date_max', 'transactionRevenue_sum'], axis=1, inplace=True)

    logger.info('Data shape is now : {}'.format(users.shape))

    return users, y, cat_indexers


def load_data(base_fold):
    df_train = pd.read_csv(f'{base_fold}/train.csv')
    df_test = pd.read_csv(f'{base_fold}/test.csv')
    df_hist_trans = pd.read_csv(f'{base_fold}/historical_transactions.csv')
    df_new_merchant_trans = pd.read_csv(f'{base_fold}/new_merchant_transactions.csv')

    return df_train, df_test, df_hist_trans, df_new_merchant_trans