import pandas as pd
import datetime


df_train = pd.read_csv("../data/churn_train.csv")
df_holdout = pd.read_csv("../data/churn_test.csv")
df_ls = [df_train, df_holdout]
for df in df_ls:
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df["churn"] = df['last_trip_date'].copy()
    churn_date = datetime.datetime(2014, 6, 1)
    churn_true = set(df[df["churn"] >= churn_date]["churn"].index)
    df.loc[churn_true, "churn"] = True
    churn_false = set(df[df["churn"] != True]["churn"].index)
    df.loc[churn_false, "churn"] = False

df_train.to_csv("../data/clean_train.csv")
df_holdout.to_csv("../data/clean_holdout.csv")