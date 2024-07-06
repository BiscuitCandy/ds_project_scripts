import pandas as pd 

df = pd.read_csv("tokenized_renal_data.csv", sep="~")

f = open("renal_data.txt", "w+")

for i in df["data"]:
    f.write(i + "\n")

print("done")