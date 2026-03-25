import pandas as pd

df = pd.read_csv("/home/tobiasfc/0arla/inputdata/train.csv")
with open("alpaca_gpt4_full.txt", "w", encoding="utf-8") as f:
    for row in df.itertuples(index=False):
        line = " ".join(str(item) for item in row if pd.notna(item))
        f.write(line + "\n")