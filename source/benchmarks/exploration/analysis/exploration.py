import os
import re
import sys
import numpy as np
import pandas as pd
import scipy.stats

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def concatCSVFiles(directory):
    contents = ""
    header = True
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            with open(os.path.join(directory, file), 'r') as fd:
                first = True
                for line in fd.readlines():
                    if not header and first:
                        first = False
                        continue
                    if header and first:
                        header = False
                        first = False
                    contents += line + "\n"
            continue
        else:
            continue
    return contents

def createPandasDataFrame(csvdata):
    return pd.read_csv(StringIO(csvdata), sep=";")

def processData(data):
    filtered = data[(data.ExplorationMode == "Full Exploration")]
    filtered = filtered[["Query", "Host", "DeviceType", "Device", "VariantTag", "Variant", "Min", "Max", "Median",
                         "Mean", "Stdev", "Var"]]
    fastest = filtered.sort_values(["Device", "Query", "Mean"]).groupby(["Device", "Query"]).head(1)
    fastest.loc[:, "VariantTag"] = fastest.loc[:, "DeviceType"] + " optimized"

    optimal = fastest

    for row in fastest.itertuples():
        selected = filtered[(filtered.Query == row.Query) & (filtered.Variant == row.Variant) &
                            (filtered.DeviceType != row.DeviceType)]

        selected.loc[:, "VariantTag"] = row.DeviceType + " optimized"

        optimal = pd.concat([optimal, selected])

        if len(selected) != len(data["Device"].unique()) - 1:
            print("Error processing the data. Maybe the csv files are corrupt!")
            sys.exit(-1)

    optimal = optimal.drop_duplicates()

    return optimal

def processDataFeatureWise(data):
    result = pd.DataFrame()
    original = data[(data.ExplorationMode == "Feature-wise Exploration") & data.VariantTag.str.match("Iteration*")]

    for device in original.DeviceType.unique():
        filtered = original[(original["DeviceType"] == device)]
        print(type(filtered))
        filteredData = pd.DataFrame()
        for query in filtered.Query.unique():
          filtered2 = filtered[(filtered["Query"] == query)]
          max_iteration = filtered2.VariantTag.max()
          filtered2 = filtered2[(data["VariantTag"] == max_iteration)]
          filtered2.loc[:, "VariantTag"] = "feature_wise"
          filteredData = filteredData.append(filtered2)
        result = pd.concat([result, filteredData])

    return result
