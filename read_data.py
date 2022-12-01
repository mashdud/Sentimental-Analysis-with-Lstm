import pandas as pd
import numpy as np

def read_data(path):
  df = pd.read_csv(path, delimiter='\t')
  df = df.rename(columns={"Unnamed: 0": "uniqueID"})
  df['date'] = pd.to_datetime(df['date'])
  return df

