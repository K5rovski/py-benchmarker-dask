from time import time
from data_train import train_model_user_lightgbm
INPUT_PATH = ''


def main_simple_ga_model():
    train_df = load_df()
    test_df = load_df("../input/test.csv")
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]


def main_user_lightgbm(sum_of_logs=False, nrows=None):
    try:
        trn_users, trn_y, indexers = get_user_data(file_name=TRN_PATH, cat_indexers=None, nrows=nrows,
                                                   sum_of_logs=sum_of_logs)
        sub_users, _, _ = get_user_data(file_name=SUB_PATH, cat_indexers=indexers, nrows=nrows)

        sub_preds, oof_preds = train_model_user_lightgbm(trn_users, trn_y, sub_users)

        # Stay in logs for submission
        sub_users['PredictedLogRevenue'] = sub_preds
        sub_users[['PredictedLogRevenue']].to_csv("simple_lgb.csv", index=True)

        logger.info('Submission data shape : {}'.format(sub_users[['PredictedLogRevenue']].shape))

        hist, bin_edges = np.histogram(np.hstack((oof_preds, sub_preds)), bins=25)
        plt.figure(figsize=(12, 7))
        plt.title('Distributions of OOF and TEST predictions', fontsize=15, fontweight='bold')
        plt.hist(oof_preds, label='OOF predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.hist(sub_preds, label='TEST predictions', alpha=.6, bins=bin_edges, density=True, log=True)
        plt.legend()
        plt.savefig('distributions.png')

    except Exception as err:
        logger.exception("Unexpected error")

def main_elo_world():
    pass

def main_elo_first_kernel():
    pass

def run_ridge_lb():
    pass
    mp.set_start_method('forkserver', True)

    start_time = time()

    train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                          engine='c',
                          dtype={'item_condition_id': 'category',
                                 'shipping': 'category'}
                          )
    test = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'),
                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'}
                         )
    print(f'[{time() - start_time}] Finished to load data')
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    submission: pd.DataFrame = test[['test_id']]

    merge, y_train, nrow_train = preprocess_pandas(train, test, start_time)

    meta_params = {'name_ngram': (1, 2),
                   'name_max_f': 75000,
                   'name_min_df': 10,

                   'category_ngram': (2, 3),
                   'category_token': '.+',
                   'category_min_df': 10,

                   'brand_min_df': 10,

                   'desc_ngram': (1, 3),
                   'desc_max_f': 150000,
                   'desc_max_df': 0.5,
                   'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
    # 'i', 'so', 'its', 'am', 'are'])

    vectorizer = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder())
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    sparse_merge = vectorizer.fit_transform(merge)
    print(f'[{time() - start_time}] Merge vectorized')
    print(sparse_merge.shape)

    tfidf_transformer = TfidfTransformer()

    X = tfidf_transformer.fit_transform(sparse_merge)
    print(f'[{time() - start_time}] TF/IDF completed')

    X_train = X[:nrow_train]
    print(X_train.shape)

    X_test = X[nrow_train:]
    del merge
    del sparse_merge
    del vectorizer
    del tfidf_transformer
    gc.collect()

    X_train, X_test = intersect_drop_columns(X_train, X_test, min_df=1)
    print(f'[{time() - start_time}] Drop only in train or test cols: {X_train.shape[1]}')
    gc.collect()

    ridge = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)
    ridge.fit(X_train, y_train)
    print(f'[{time() - start_time}] Train Ridge completed. Iterations: {ridge.n_iter_}')

    predsR = ridge.predict(X_test)
    print(f'[{time() - start_time}] Predict Ridge completed.')