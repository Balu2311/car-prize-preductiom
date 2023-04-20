import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from sklearn.metrics import mean_squared_error,mean_absolute_error
#%matplotlib inline

df = pd.read_csv('car data.csv')
print(df.head())

print(df.shape)

print(df.dtypes[df.dtypes == object])

print(df.describe())

print(df.info())

for col in df.dtypes[df.dtypes == object].index:
    print('Unique items in column',col,'are:',df[col].unique())
    print('-'*75)

for col in df[['Fuel_Type','Seller_Type','Transmission']]:
    print(df[col].value_counts())
    print('-'*75)

print(df['Owner'].unique())

print(df.isnull().sum())

df.drop(['Car_Name'],axis = 1,inplace = True)
print(df.head())

df['Current_Year'] = 2022
df['Number_Of_Years'] = df['Current_Year'] - df['Year']

df.drop(['Year','Current_Year'],axis = 1,inplace = True)

print(df.head())

print(df['Transmission'].value_counts())

def Encode(df,variable):
    encoded_Variable = df[variable].value_counts().to_dict()
    df[variable] = df[variable].map(encoded_Variable)

for col in df[['Fuel_Type','Seller_Type','Transmission']]:
    Encode(df,col)
print(df.head())

sns.pairplot(df)

df_corr = df.corr()
index = df_corr.index

plt.figure(figsize = (12,6))
sns.heatmap(df[index].corr(),annot = True,cmap = 'rainbow')


X = df.drop(['Selling_Price'],axis = 1)
y = df['Selling_Price']

print(X.head())

print(y.head())

threshold = .52

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

correlation(X,threshold)

model=ExtraTreesRegressor(random_state = 101)
model.fit(X,y)

print(model.feature_importances_)

pd.Series(model.feature_importances_, index = X.columns).nlargest(5).plot(kind = 'barh')

rf = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

print(rf_random.fit(X_train,y_train))


y_pred = rf_random.predict(X_test)


sns.histplot(y_test - y_pred,kde = True)

plt.scatter(y_test,y_pred)


param_grid = {
    'max_depth': [rf_random.best_params_['max_depth']],
    'max_features': [rf_random.best_params_['max_features']],
    'min_samples_leaf': [rf_random.best_params_['min_samples_leaf'],
                         rf_random.best_params_['min_samples_leaf']+2],
    'min_samples_split': [rf_random.best_params_['min_samples_split'] - 2,
                          rf_random.best_params_['min_samples_split'] - 1,
                          rf_random.best_params_['min_samples_split'],
                          rf_random.best_params_['min_samples_split'] +1,
                          rf_random.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_random.best_params_['n_estimators'] - 100,
                     rf_random.best_params_['n_estimators'],
                     rf_random.best_params_['n_estimators'] + 100]
}

print(param_grid)

rf=RandomForestRegressor()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
grid_search.fit(X_train,y_train)


print(grid_search)

y_pred=grid_search.predict(X_test)
sns.histplot(y_test - y_pred, kde = True)

plt.scatter(y_test, y_pred)


params = {'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200])
    }
print(params)

seed=2
def objective(params):
    est = int(params['n_estimators'])
    md = int(params['max_depth'])
    msl = int(params['min_samples_leaf'])
    mss = int(params['min_samples_split'])
    model = RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss)
    model.fit(X_train,y_train)
    y_pred_hyperopt = model.predict(X_test)
    score = mean_squared_error(y_test,y_pred_hyperopt)
    return score

def optimize(trial):
    params={'n_estimators':hp.uniform('n_estimators',100,500),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,6)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=100,rstate=np.random.RandomState(seed))
    return best

trial=Trials()
best=optimize(trial)

print(best)


for t in trial.trials[:2]:
    print (t)

TID = [t['tid'] for t in trial.trials]
Loss = [t['result']['loss'] for t in trial.trials]
maxd = [t['misc']['vals']['max_depth'][0] for t in trial.trials]
nest = [t['misc']['vals']['n_estimators'][0] for t in trial.trials]
min_ss = [t['misc']['vals']['min_samples_split'][0] for t in trial.trials]
min_sl = [t['misc']['vals']['min_samples_leaf'][0] for t in trial.trials]

hyperopt_rfr = pd.DataFrame({'tid':TID,'loss':Loss,
                          'max_depth':maxd,'n_estimators':nest,
                          'min_samples_split':min_ss, 'min_samples_leaf':min_sl})
hyperopt_rfr.head()

trainedforest = RandomForestRegressor(max_depth = best['max_depth'],
                                       min_samples_leaf = round(best['min_samples_leaf']),
                                       min_samples_split = round(best['min_samples_split']),
                                       n_estimators = int(best['n_estimators'])).fit(X_train,y_train)

y_pred_hyperopt = trainedforest.predict(X_test)

sns.histplot(y_test - y_pred_hyperopt,kde = True)

plt.scatter(y_test,y_pred_hyperopt)
print('MAE is:',mean_absolute_error(y_test,y_pred_hyperopt))
print('MSE is:',mean_squared_error(y_test,y_pred_hyperopt))
print('RMSE is:',np.sqrt(mean_squared_error(y_test,y_pred_hyperopt)))

import pickle
file = open('regression_rf_pycham.pkl','wb')
pickle.dump(trainedforest,file)
