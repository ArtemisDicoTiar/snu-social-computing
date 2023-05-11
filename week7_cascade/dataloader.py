import pandas as pd

from week6_pagerank.pagerank_mapreduce import DATA_DIR

COVID_DIR = DATA_DIR / "covid"

covid_df = pd.read_csv(COVID_DIR / "countries-aggregated.csv")
refer_df = pd.read_csv(COVID_DIR / "reference.csv")

korea_df = covid_df[covid_df["Country"] == "Korea, South"].drop(["Country"], axis=1)
us_df = covid_df[covid_df["Country"] == "US"].drop(["Country"], axis=1)
uk_df = covid_df[covid_df["Country"] == "United Kingdom"].drop(["Country"], axis=1)
global_df = covid_df.groupby("Date").sum()

korea_pop = refer_df[refer_df["Combined_Key"] == "Korea, South"]["Population"].values.tolist()[0]
us_pop = refer_df[refer_df["Combined_Key"] == "US"]["Population"].values.tolist()[0]
uk_pop = refer_df[refer_df["Combined_Key"] == "United Kingdom"]["Population"].values.tolist()[0]
global_pop = refer_df[refer_df["Province_State"].isnull()]["Population"].sum()

print()
