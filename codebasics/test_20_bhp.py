import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

def exercise_1(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    # logger.info(f"Dataset shape: {df.shape}")
    # logger.info(f"Dataset description: {df.describe()}")
    # logger.info(f"Feature names: {df.columns.tolist()}")
    # logger.info(f"First 5 rows :\n{df.head()}")
    # logger.info(f'{df.isnull().sum()}')

    df2 = df.drop(['area_type','society','balcony','availability'],axis='columns')
    df3 = df2.dropna()
    # df4 = df3.copy
    # df3.loc[:, 'bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
    logger.info(f'bhk unique: {df3.bhk.unique()}')

    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True
    logger.info(f'ddd: {df3[~df3['total_sqft'].apply(is_float)].head(10)}')

    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0])+float(tokens[1]))/2
        try:
            return float(x)
        except:
            return None   
    df4 = df3.copy()
    df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
    df4 = df4[df4.total_sqft.notnull()]
    
    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']

    df5.location = df5.location.apply(lambda x: x.strip())
    location_stats = df5['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats<=10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    df6 = df5[~(df5.total_sqft/df5.bhk<300)]
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
            df_out = pd.concat([df_out,reduced_df],ignore_index=True)
        return df_out
    df7 = remove_pps_outliers(df6)

    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count']>5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
        return df.drop(exclude_indices,axis='index')
    df8 = remove_bhk_outliers(df7)
    df9 = df8[df8.bath<df8.bhk+2]
    
    df10 = df9.drop(['size','price_per_sqft'],axis='columns')
    dummies = pd.get_dummies(df10.location)
    df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
    df12 = df11.drop('location',axis='columns')
    df12.to_csv("bhp.csv",index=False)

def exercise_2(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    # logger.info(f"Dataset shape: {df.shape}")
    # logger.info(f"Dataset description: {df.describe()}")
    # logger.info(f"Feature names: {df.columns.tolist()}")
    # logger.info(f"First 5 rows :\n{df.head()}")

    X = df.drop(['price'],axis='columns')
    y = df.price

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

    from sklearn.linear_model import LinearRegression
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    logger.info(f'lr score: {lr_clf.score(X_test,y_test)}')

    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    logger.info(f'kfold score: {cross_val_score(LinearRegression(), X, y, cv=cv)}')

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso
    from sklearn.tree import DecisionTreeRegressor
    def find_best_model_using_gridsearchcv(X,y):
        algos = {
            'linear_regression' : {
                'model': LinearRegression(),
                'params': {
                    'normalize': [True, False]
                }
            },
            'lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [1,2],
                    'selection': ['random', 'cyclic']
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'criterion' : ['mse','friedman_mse'],
                    'splitter': ['best','random']
                }
            }
        }
        scores = []
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        for algo_name, config in algos.items():
            gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
            gs.fit(X,y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })

        return pd.DataFrame(scores,columns=['model','best_score','best_params'])

    # logger.info(f'score: {find_best_model_using_gridsearchcv(X,y)}')
    # print(find_best_model_using_gridsearchcv(X,y))

    def predict_price(location,sqft,bath,bhk):    
        loc_index = np.where(X.columns==location)[0][0]

        x = np.zeros(len(X.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        return lr_clf.predict([x])[0]

    logger.info(f'predict value: {predict_price('1st Phase JP Nagar',1000, 2, 2)}')

    import pickle
    with open('banglore_home_prices_model.pickle','wb') as f:
        pickle.dump(lr_clf,f)

    import json
    columns = {
        'data_columns' : [col.lower() for col in X.columns]
    }
    with open("columns.json","w") as f:
        f.write(json.dumps(columns))

        
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Run exercises for Banglor House Pricing',
        epilog='Example usage: python test_20_bhp.py 1',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument('--exercise', type=int, default=1, help='Exercise number to run')
    argparser.add_argument('-f', '--file', type=str, default=None, help='Verbose output')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(filename='test_20_bhp.log', filemode='a', level= log_level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    action = {
        1: lambda: exercise_1(args.file),
        2: lambda: exercise_2(args.file),
    }
    if args.exercise in action:
        logger.info(f"Running exercise {args.exercise}")
        action[args.exercise]()
    else:
        logger.warning(f"Exercise {args.exercise} not defined")

    logger.info("=" * 20)
    logger.info("Script execution completed. \n"+"=" * 50)