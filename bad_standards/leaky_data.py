import pandas as pd
import glob

def load_everything():
    all_data = []
    # This might crash if there are many csvs, no error handling
    for f in glob.glob("data/*.csv"):
        df = pd.read_csv(f)
        all_data.append(df)
    
    # Returning a huge list of dataframes without joining properly or cleaning
    return all_data

x = load_everything()
print(x)
