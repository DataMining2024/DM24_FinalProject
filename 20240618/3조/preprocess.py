import pandas as pd
import numpy as np



df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

output_columns = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
output_sd = ["X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]
other_columns = df.columns.drop([*output_columns, *output_sd, 'id'])

# filter outliers based on >=0 and 99th quantile
for column in output_columns:
    upper_quantile = df[column].quantile(0.99)
    df = df[(df[column] < upper_quantile) & (df[column] >= 0)]
 
 

# get correlation matrix
correlation_matrix = df.corr()

def get_sorted_correlation(cm, column_name):
    return cm[column_name].drop(output_sd+output_columns).abs().sort_values(ascending=False)


# get sorted data based on correlation on output columns
d = {}
for cn in output_columns:
    d[cn] = [(k, v) for (k, v) in get_sorted_correlation(correlation_matrix, cn).items()]

# Extract top-n features for each label
finald = {}
percentages = [10, 20, 30, 40, 50, 75]
for cn in output_columns:
    finald[cn] = {}
    for p in percentages:
        finald[cn][p] = d[cn][:round(len(d[cn])*(p/100))]

# Save the extracted features into .csv
for output_name in finald:
    for percent in finald[output_name]:
        cur = finald[output_name][percent]
        feature_names = [x for (x, y) in cur]

        cur_df = df[["id", *feature_names, *output_columns]]
        cur_df.to_csv(f"top/{output_name}_{percent}.csv")

        cur_test_df = test_df[["id", *feature_names]]
        cur_test_df.to_csv(f"test_top/{output_name}_{percent}.csv")
        
        
        
# Extract top-n features for all labels combined (naive sum of correlation factors)
combined = {}
for output_name in d:
    for feature_name, correlation_factor in d[output_name]:
        combined[feature_name] = combined.get(feature_name, 0) + correlation_factor
combined = sorted(list(combined.items()), key=lambda x: -x[1])

# Save top-n featues for all labels
for p in percentages:
    feature_names = [x for (x, y) in combined[:round(len(combined)*(p/100))]]
    cur_df = df[["id", *feature_names, *output_columns]]
    cur_df.to_csv(f"top/all_{p}.csv")

    cur_test_df = test_df[["id", *feature_names]]
    cur_test_df.to_csv(f"test_top/all_{p}.csv")
    print(feature_names)