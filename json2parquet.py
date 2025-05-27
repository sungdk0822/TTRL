# https://stackoverflow.com/a/68229753
import pandas as pd
data = pd.read_json('/home/elicer/TTRL/verl/data/AIME-TTT/test.json')
print(data)
data.to_parquet('/home/elicer/TTRL/verl/data/AIME-TTT/test.parquet')