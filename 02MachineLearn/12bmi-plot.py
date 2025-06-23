
import matplotlib.pyplot as plt

import pandas as pd
# 세번째 열을 행인덱스로 지정함
tbl = pd.read_csv("./resData/bmi.csv", index_col=2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def scatter (lbl, color):
    b= tbl.loc[lbl]
    ax.scattter(b["weight"],b["height"], c=color, label=lbl)

    scatter("fay", "red")
    scatter("normal", "yellow")
    scatter("thin", "purple")

    ax.legend()
    plt.savefig("./saveFiles/bmi-scatter.png")
    plt.show()