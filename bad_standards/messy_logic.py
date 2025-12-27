import pandas as pd
import os

def do_stuff_to_csv(f):
    # bad name, no docstring
    d = pd.read_csv(f)
    for i in range(len(d)):
        # slow loop
        if d.iloc[i]['amount'] > 1000:
            print("BIG")
        else:
            print("SMALL")
    d.to_csv("temp_file.csv") # hardcoded path

do_stuff_to_csv("data/transactions.csv")
