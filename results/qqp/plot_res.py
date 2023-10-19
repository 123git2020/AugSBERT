
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

mpl.style.use("ggplot")
mpl.rcParams["lines.markerfacecolor"]="None"
mpl.rcParams["axes.titlecolor"]="#555555"
plt.rcParams['xtick.labelsize']=12
plt.rcParams["axes.prop_cycle"]=(cycler(color=['#E24A33', '#988ED5', '#777777']))


df_in=pd.read_csv("in_domain.csv",header=0)
df_cross=pd.read_csv("cross_domain_aug.csv",header=0)
df_no=pd.read_csv("no_train.csv",header=0)

df_in=df_in[["cossim_f1","manhattan_f1","euclidean_f1","dot_f1"]]
df_cross=(df_cross[["cossim_f1","manhattan_f1","euclidean_f1","dot_f1"]]).dropna()
df_no=df_no[["cossim_f1","manhattan_f1","euclidean_f1","dot_f1"]]

in_=list(df_in.mean())
cross_=list(df_cross.mean())
no_=list(df_no.mean())

name=[n[:-3] for n in df_no.columns]



x=np.arange(4)
width=0.2
multiplier=0

plt.figure(figsize=(8,5),dpi=150)

for n, v in zip(["no_train","cross_domain","in_domain"], [no_,cross_,in_]):
    offset = width * multiplier
    m=plt.bar(x + offset, np.array(v), width, label=n,alpha=0.7,bottom=0)
    plt.bar_label(m,fmt='%.3f',padding=3,fontsize=6,color="#555555")
    multiplier += 1

plt.ylabel('F1 score')
plt.title('F1 score with different measurements')
plt.xticks(x+width, name)


plt.legend(loc='upper left', ncols=3, framealpha=0.3, edgecolor="None",labelcolor="#555555",fontsize=9)
plt.ylim(top=0.9)

plt.show()