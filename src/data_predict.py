from sklearn.metrics import mean_squared_log_error



def predict_ga_simple(x,y, y_pred, model, xtest):
    # # Evaluation
    model = CustomRegressor(n_jobs=-1, verbose=1)
    model.fit(x, y)
    y_pred = pd.DataFrame(model.predict(x),index=x.index,columns=y.columns)
    # In[35]:

    y_pred = y_pred.stack(['store_nbr', 'family']).clip(0.)
    y_ = y.stack(['store_nbr', 'family']).clip(0.)

    y_['pred'] = y_pred.values
    print(y_.groupby('family').apply(lambda r: np.sqrt(np.sqrt(mean_squared_log_error(r['sales'], r['pred'])))))
    # Looking at error
    print('RMSLE : ', np.sqrt(np.sqrt(msle(y_['sales'], y_['pred']))))

    # All seems good.

    # In[36]:

    y_pred.isna().sum()

    # In[37]:

    ypred = pd.DataFrame(model.predict(xtest), index=xtest.index, columns=y.columns).clip(0.)
    ypred

    # In[38]:

    ypred = ypred.stack(['store_nbr', 'family'])
    ypred

