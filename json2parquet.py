# https://stackoverflow.com/a/68229753
import pandas as pd
data = pd.read_json('/home/elicer/TTRL/verl/data/AIME-TTT/train.json')
print(data)
data.to_parquet('/home/elicer/TTRL/verl/data/AIME-TTT/train.parquet')