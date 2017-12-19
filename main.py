'''
main.py
Entry point
Ankur Goswami, agoswam3@ucsc.edu
'''

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trainf = 'data/train.csv'

def mergecols(df, newcol, col1, col2):
    df[newcol] = df[col1].map(str) + df[col2].map(str)
    return df

def main():
    df = pd.read_csv(trainf)
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    vals = []
    for column in columns:
        countdf = df[column].value_counts()
        vals.append(countdf[1])

    plt.bar(columns, vals)
    plt.show()
    merged_columns = []
    for columnx in columns:
        for columny in columns:
            if columnx == columny or (columny + '+' + columnx) in merged_columns:
                continue
            merge = columnx + '+' + columny
            merged_columns.append(merge)
            df = mergecols(df, merge, columnx, columny)
    
    vals1 = []
    for mergedc in merged_columns:
        countdf = df[mergedc].value_counts()
        vals1.append(countdf['11'])

    for i, val in enumerate(merged_columns):
        spl = val.split('+')
        x = [val, spl[0], spl[1]]
        y = [vals1[i], vals[columns.index(spl[0])], vals[columns.index(spl[1])]]
        plt.bar(x, y)
        plt.show()

main()
