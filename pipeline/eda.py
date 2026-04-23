import pandas as pd

def basic_eda(df):
    results = {}

    results["shape"] = df.shape
    results["columns"] = list(df.columns)
    results["dtypes"] = df.dtypes.astype(str).to_dict()
    results["missing_values"] = df.isnull().sum().to_dict()
    results["describe"] = df.describe(include="all").to_dict()

    return results