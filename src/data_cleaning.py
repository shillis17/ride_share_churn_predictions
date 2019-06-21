import pandas as pd
import datetime


df_train = pd.read_csv("../data/churn_train.csv")
df_holdout = pd.read_csv("../data/churn_test.csv")
df_ls = [df_train, df_holdout]
for df in df_ls:
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df["churn"] = df['last_trip_date'].copy()
    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)
    churn_date = datetime.datetime(2014, 6, 1)
    phone_true = set(df[df["phone"] == "iPhone"]["phone"].index)
    df.loc[phone_true, "phone"] = 1
    phone_false = set(df[df["phone"] != 1]["phone"].index)
    df.loc[phone_false, "phone"] = 0
    churn_true = set(df[df["churn"] >= churn_date]["churn"].index)
    df.loc[churn_true, "churn"] = True
    churn_false = set(df[df["churn"] != True]["churn"].index)
    df.loc[churn_false, "churn"] = False

df_train.to_csv("../data/clean_train.csv")
df_holdout.to_csv("../data/clean_holdout.csv")