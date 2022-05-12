import pandas as pd

df = pd.read_csv('./results/DEBUG_results_2env.csv', sep=', ', engine='python')

print(
    df[
        (df['Soft'] == False)
        & (df['Number of environments'] == df['Number of environments'].max())
    ].groupby(
        ['Method']
    ).mean()[['Precision', 'Recall', 'Average precision']]
)
