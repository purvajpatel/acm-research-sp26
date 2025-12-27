import pandas as pd

df = pd.read_parquet("hf://datasets/nyu-mll/glue/ax/test-00001-of-00002.parquet")

print(df.shape[0])