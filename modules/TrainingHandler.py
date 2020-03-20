import pickle
import random
import csv
import itertools
from collections import defaultdict
from os import path
from operator import add

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.contrib.statsmodels.base import StatsModelsWrapper

from modules.MyDatabase import MyDatabase

COLUMNS = {"winequality_red": ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                               'ph', 'sulphates', 'alcohol'],
           "houseprice": ['price', 'bedrooms', 'bathrooms', 'sqft_living',
                          'sqft_lot', 'floors', 'waterfront', 'condition', 'grade',
                          'yr_built', 'zipcode']}

TARGETS = {"winequality_red": "quality",
           "car_price_prediction": "price",
           "youtube-new": "views"}

PATHS = {"winequality_red": "data/winequality_red/winequality_red.csv",
         "houseprice": "data/houseprice/kc_house_data.csv"}

# TASKS = ["car_price_prediction", "winequality_red", "youtube-new"]
TASKS = ["houseprice"]

# SCHEMAS = ["train", "0.1", "0.2", "0.3", "0.4", "0.5"]
SCHEMAS = ["train"] + [str(i) for i in range(1, 33)]

SEEDS = [0, 1, 2, 3, 4]


def cross_validation(k, formula, d, target, evaluation):
    indices = list(range(d.shape[0]))

    model_scores = defaultdict()

    for c in range(k):
        trainset = set(indices)
        random.shuffle(indices)
        testset = {indices[i] for i in range(len(indices)) if i % k == 0}
        trainset = trainset - testset
        train = d.iloc[list(trainset)]
        test = d.iloc[list(testset)]

        m = smf.ols(formula=formula,
                    data=train)

        res = m.fit()

        model_scores[c] = {'mse': getPerformance(res, test, target)[0],
                           'model': res}
    return model_scores[min(model_scores, key=(lambda x: model_scores[x][evaluation]))]['model']


def getPerformance(m, df, target):
    yHat = m.predict(df)
    y = df[target].values
    return mean_squared_error(y, yHat), mean_absolute_error(y, yHat), m.rsquared, m.rsquared_adj


def holdOutSplit(task, seed):
    df = pd.read_csv(PATHS.get(task),usecols=COLUMNS.get(task))

    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    holdOut_df = df[~msk]

    train_df.to_csv("data/{}/preinput/{}_seed{}_train.csv".format(task, task, seed), index=False, header=False, quoting=csv.QUOTE_NONE)
    holdOut_df.to_csv("data/{}/{}_seed{}_holdout.csv".format(task, task, seed), index=False, header=False, quoting=csv.QUOTE_NONE)


def plotScatters(df, task, schema):
    pd.plotting.scatter_matrix(df, figsize=(15, 15))
    plt.title("{} ,schema: {}".format(task, schema))
    plt.savefig("figs/scatters_{}_seed{}_{}.png".format(task, seed, schema))
    plt.close()


def plotResiduals(df, holdOut_df, task, schema, seed, model):
    y_train = df[TARGETS.get(task)].values
    y_trainHat = model.predict(df)

    y_test = holdOut_df[TARGETS.get(task)].values
    y_testHat = model.predict(holdOut_df)

    data = []
    for v in range(11):
        data.append([y - yh for y, yh in zip(y_train, y_trainHat) if y == v])
    plt.subplot(2, 1, 1)
    plt.boxplot(data)

    data = []
    for v in range(11):
        data.append([y - yh for y, yh in zip(y_test, y_testHat) if y == v])
    plt.subplot(2, 1, 2)
    plt.boxplot(data)
    plt.savefig("figs/box_residuals_{}_seed{}_{}.png".format(task, seed, schema))
    plt.close()


def plotResidualsAgainstHoldout(df, holdOut_df, task, seed, schema):
    X_train = df[COLUMNS.get(task)].values
    X_test = holdOut_df[COLUMNS.get(task)].values
    y_train = df[TARGETS.get(task)].values
    y_test = holdOut_df[TARGETS.get(task)].values

    # Instantiate the linear model and visualizer
    wrapped_model = LinearRegression()
    visualizer = ResidualsPlot(wrapped_model, title="Residuals for schema {}".format(schema))

    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show(outpath="figs/residuals_{}_seed{}_{}.png".format(task, seed, schema))
    plt.close()


def plotPredictedAgainstActual(df, holdOut_df, task, seed, schema, model):
    y_train = df[TARGETS.get(task)].values
    y_trainHat = model.predict(df)

    y_test = holdOut_df[TARGETS.get(task)].values
    y_testHat = model.predict(holdOut_df)

    # import pdb
    # pdb.set_trace()
    plt.subplot(2, 1, 1)
    plt.scatter(y_train, y_trainHat)
    plt.title('Predicted v.s Actual - Training')
    plt.ylabel('Predicted')
    plt.ylim((0, 10))
    plt.xlim((0, 10))

    plt.subplot(2, 1, 2)
    plt.scatter(y_test, y_testHat)
    plt.title('Predicted v.s Actual - Holdout')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.ylim((0, 10))
    plt.xlim((0, 10))

    plt.savefig("figs/fitted_actual_{}_seed{}_{}.png".format(task, seed, schema))
    plt.close()


def getData(db, task, seed, schema):
    if schema in ["train", "holdout"]:
        sql = """
        select * from {}_seed{}_{}
        """.format(task, seed, schema)
    else:
        sql = """
        select * from {}_seed{}_train_v_{}
        """.format(task, seed, schema)
    df = pd.DataFrame([i for i in db.getSQLResult(sql)])
    df = df[COLUMNS.get(task) + [TARGETS.get(task)]]
    return df.apply(pd.to_numeric)


def getSchemaStats(db, task, seed, schema):
    if schema in ["train", "holdout"]:
        sql = """select count(*) as c from {}_seed{}_{}""".format(task, seed, schema)
        totalNumberOfRows = [i for i in db.getSQLResult(sql)][0]['c']

        sql = """select count(*) as c from information_schema.columns where table_catalog = 'cse544' and table_name = '{}_seed{}_{}' """.format(task, seed, schema)
        numberOfColumns = [i for i in db.getSQLResult(sql)][0]['c']

        totalNumberOfCells = numberOfColumns*totalNumberOfRows

    
    else:
        sql = """select count(*) as c from {}_seed{}_train_v_{}""".format(task, seed, schema)
        totalNumberOfRows = [i for i in db.getSQLResult(sql)][0]['c']
        
        sql = """
        select table_name FROM information_schema.tables where table_name like '{}_seed{}_train_{}\_%'
        """.format(task, seed, schema)
        tables = [i['table_name'] for i in db.getSQLResult(sql)]

        totalNumberOfCells = 0
        for table in tables:
            sql = """select count(*) as c from information_schema.columns where table_catalog = 'cse544' and table_name = '{}' """.format(table)
            numberOfColumns = [i for i in db.getSQLResult(sql)][0]['c']
            
            sql = """select count(*) as c from {}""".format(table)
            numberOfRows = [i for i in db.getSQLResult(sql)][0]['c']

            totalNumberOfCells += numberOfColumns*numberOfRows
        
    return totalNumberOfRows, totalNumberOfCells


def main(task, seed):
    db = MyDatabase()

    if path.exists("data/{}/preinput/{}_seed{}_train.csv".format(task, task, seed)) and\
       path.exists("data/{}/{}_seed{}_holdout.csv".format(task, task, seed)):
        print("split already done for {}".format(task))
    else:
        print("splitting for {}, seed: {}\n".format(task, seed))
        holdOutSplit(task, seed)

    holdOut_df = getData(db, task, seed, "holdout")

    formula = "{}~{}".format(TARGETS.get(task), "+".join(COLUMNS.get(task)))

    csv_data = []
    for schema in SCHEMAS:
        try:
            data = getData(db, task, seed, schema)

            if path.exists("pickled/{}_seed{}_{}.pickle".format(task, seed, schema)):
                print("loading pickled model for {}, seed {}, schema {}".format(task, seed, schema))
                with open("pickled/{}_seed{}_{}.pickle".format(task, seed, schema), "rb") as f:
                    model = pickle.load(f)
            else:
                with open("pickled/{}_seed{}_{}.pickle".format(task, seed, schema), 'wb') as f:
                    model = cross_validation(5, formula, data, TARGETS.get(task), 'mse')
                    print("dumping pickled model for {}, seed {}, schema {}\n".format(task, seed, schema))
                    pickle.dump(model, f)

            mse, mae, r_squared, r_squared_adj = getPerformance(model, holdOut_df, TARGETS.get(task))
            numberOfRows, numerOfCells = getSchemaStats(db, task, seed, schema)
            plotResidualsAgainstHoldout(data, holdOut_df, task, seed, schema)

            with open("logs/{}_seed{}_schema{}_summary.csv".format(task, seed, schema), 'w') as f:
                f.write(model.summary().as_csv())
            print("Task: {}, seed: {}, schema: {}, MSE: {}, R-squared: {}, Adj. R-squared: {}, numberOfRows:{}, numerOfCells:{}".format(task, seed, schema, mse, r_squared, r_squared_adj, numberOfRows, numerOfCells))
            csv_data.append([task, seed, schema, mse, r_squared, r_squared_adj, numberOfRows, numerOfCells])

        except:
            pass

    with open("logs/{}_seed{}.csv".format(task, seed), 'w', newline='') as f:
        spamwriter = csv.writer(f)
        spamwriter.writerow(["Task", "Seed", "Schema", "MSE", "R-Squared", "Adj. R-Squared", "numberOfRows","numerOfCells"])
        spamwriter.writerows(csv_data)
    db.close()


def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    for seed in SEEDS:
        setSeeds(seed)
        for task in TASKS:
            #main(task, seed)
            holdOutSplit(task,seed)
