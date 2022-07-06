import time

import numpy as np
import pandas as pd


def train_simple_ga():
    lnr = LinearRegression(fit_intercept=True, n_jobs=-1, normalize=True)
    lnr.fit(x, y)

    yfit_lnr = pd.DataFrame(lnr.predict(x), index=x.index, columns=y.columns).clip(0.)
    ypred_lnr = pd.DataFrame(lnr.predict(xtest), index=xtest.index, columns=y.columns).clip(0.)

    svr = MultiOutputRegressor(SVR(C=0.2, kernel='rbf'), n_jobs=-1)
    svr.fit(x, y)

    yfit_svr = pd.DataFrame(svr.predict(x), index=x.index, columns=y.columns).clip(0.)
    ypred_svr = pd.DataFrame(svr.predict(xtest), index=xtest.index, columns=y.columns).clip(0.)

    yfit_mean = pd.DataFrame(np.mean([yfit_svr.values, yfit_lnr.values], axis=0), index=x.index,
                             columns=y.columns).clip(0.)
    ypred_mean = pd.DataFrame(np.mean([ypred_lnr.values, ypred_svr.values], axis=0), index=xtest.index,
                              columns=y.columns).clip(0.)

    y_ = y.stack(['store_nbr', 'family'])
    y_['lnr'] = yfit_lnr.stack(['store_nbr', 'family'])['sales']
    y_['svr'] = yfit_svr.stack(['store_nbr', 'family'])['sales']
    y_['mean'] = yfit_mean.stack(['store_nbr', 'family'])['sales']

    print('=' * 70, 'Linear Regression', '=' * 70)
    print(y_.groupby('family').apply(lambda r: np.sqrt(msle(r['sales'], r['lnr']))))
    print('LNR RMSLE :', np.sqrt(msle(y, yfit_lnr)))
    print('=' * 70, 'SVR', '=' * 70)
    print(y_.groupby('family').apply(lambda r: np.sqrt(msle(r['sales'], r['svr']))))
    print('SVR RMSLE :', np.sqrt(msle(y, yfit_svr)))
    print('=' * 70, 'Mean', '=' * 70)
    print(y_.groupby('family').apply(lambda r: np.sqrt(msle(r['sales'], r['mean']))))
    print('Mean RMSLE :', np.sqrt(msle(y, yfit_mean)))

    # In[21]:

    from sklearn.metrics import mean_absolute_error as mae

    print('=' * 70, 'Linear Regression', '=' * 70)
    print(y_.groupby('family').apply(lambda r: mae(r['sales'], r['lnr'])))
    print('LNR RMSLE :', mae(y, yfit_lnr))
    print('=' * 70, 'SVR', '=' * 70)
    print(y_.groupby('family').apply(lambda r: mae(r['sales'], r['svr'])))
    print('SVR RMSLE :', mae(y, yfit_svr))
    print('=' * 70, 'Mean', '=' * 70)
    print(y_.groupby('family').apply(lambda r: mae(r['sales'], r['mean'])))
    print('Mean RMSLE :', mae(y, yfit_mean))

    # As you can see, with RMSLE, the best model is the averaging of linear regression and SVR.
    #
    # But, in MAE, Linear Regression has the least loss than Mean. What does it mean?

    # Because in RMSLE we are applying log, that means higher the value, the lower the deviation.
    #
    # Let me show you

    # In[22]:

    true_low = [2]
    pred_low = [4]

    print('RMSLE for low value :', np.sqrt(msle(true_low, pred_low)))
    print('MAE for low value :', mae(true_low, pred_low))

    true_high = [255]
    pred_high = [269]

    print('RMSLE for high value :', np.sqrt(msle(true_high, pred_high)))
    print('MAE for high value :', mae(true_high, pred_high))


def train_model_user_lightgbm(trn_users, trn_y, sub_users):
    folds = KFold(n_splits=5, shuffle=True, random_state=7956112)

    sub_preds = np.zeros(sub_users.shape[0])
    oof_preds = np.zeros(trn_users.shape[0])
    oof_scores = []
    lgb_params = {
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'num_leaves': 128,
        'subsample': 0.2217,
        'colsample_bytree': 0.6810,
        'min_split_gain': np.power(10.0, -4.9380),
        'reg_alpha': np.power(10.0, -3.2454),
        'reg_lambda': np.power(10.0, -4.8571),
        'min_child_weight': np.power(10.0, 2),
        'silent': True
    }

    for fold_, (trn_, val_) in enumerate(folds.split(trn_users)):
        model = lgb.LGBMRegressor(**lgb_params)

        model.fit(
            trn_users.iloc[trn_], trn_y.iloc[trn_],
            eval_set=[(trn_users.iloc[trn_], trn_y.iloc[trn_]),
                      (trn_users.iloc[val_], trn_y.iloc[val_])],
            eval_metric='rmse',
            early_stopping_rounds=100,
            verbose=0
        )

        oof_preds[val_] = model.predict(trn_users.iloc[val_])
        curr_sub_preds = model.predict(sub_users)
        curr_sub_preds[curr_sub_preds < 0] = 0
        sub_preds += curr_sub_preds / folds.n_splits
        #     preds[preds <0] = 0

        logger.info('Fold %d RMSE (raw output) : %.5f' % (fold_ + 1, rmse(trn_y.iloc[val_], oof_preds[val_])))
        oof_preds[oof_preds < 0] = 0
        oof_scores.append(rmse(trn_y.iloc[val_], oof_preds[val_]))
        logger.info('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))

    logger.info('Full OOF RMSE (zero clipped): %.5f +/- %.5f' % (rmse(trn_y, oof_preds), float(np.std(oof_scores))))
    return sub_preds, oof_preds

def train_first_kernel(df_train, df_test, df_train_columns, target):
    pass
    param = {'num_leaves': 31,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.1,
             "verbosity": -1,
             "nthread": 4,
             "random_state": 4590}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
    oof = np.zeros(len(df_train))
    predictions = np.zeros(len(df_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, df_train['outliers'].values)):
        print("fold {}".format(fold_))
        trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns],
                               label=target.iloc[trn_idx])  # , categorical_feature=categorical_feats)
        val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns],
                               label=target.iloc[val_idx])  # , categorical_feature=categorical_feats)

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                        early_stopping_rounds=100)
        oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = df_train_columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    mse = np.sqrt(mean_squared_error(oof, target))

    return predictions,feature_importance_df,mse


def data_train_elo_world(train, target, test):
    pass

    features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
    # features = [f for f in features if f not in unimportant_features]
    categorical_feats = ['feature_2', 'feature_3']

    # We then set the hyperparameters of the LGBM model, these parameters are obtained by an [bayesian optimization done in another kernel](https://www.kaggle.com/fabiendaniel/hyperparameter-tuning/edit):

    # In[ ]:

    param = {'num_leaves': 111,
             'min_data_in_leaf': 149,
             'objective': 'regression',
             'max_depth': 9,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.7522,
             "bagging_freq": 1,
             "bagging_fraction": 0.7083,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.2634,
             "random_state": 133,
             "verbosity": -1}

    # We now train the model. Here, we use a standard KFold split of the dataset in order to validate the results and to stop the training. Interstingly, during the writing of this kernel, the model was enriched adding new features, which improved the CV score. **The variations observed on the CV were found to be quite similar to the variations on the LB**: it seems that the current competition won't give us headaches to define the correct validation scheme:

    # In[ ]:

    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    start = time.time()
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                               label=target.iloc[trn_idx],
                               categorical_feature=categorical_feats
                               )
        val_data = lgb.Dataset(train.iloc[val_idx][features],
                               label=target.iloc[val_idx],
                               categorical_feature=categorical_feats
                               )

        num_round = 10000
        clf = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=100,
                        early_stopping_rounds=200)

        oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target) ** 0.5))
    return feature_importance_df