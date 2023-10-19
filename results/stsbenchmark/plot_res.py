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


df_aug=pd.read_csv("similarity_test_aug.csv",header=0)

df_sbert=pd.read_csv("similarity_test_sbert.csv",header=0)

df_aug=df_aug[["cosine_spearman","manhattan_spearman","euclidean_spearman","dot_spearman"]]
df_sbert=df_sbert[["cosine_spearman","manhattan_spearman","euclidean_spearman","dot_spearman"]]

aug=list(df_aug.mean())
sbert=list(df_sbert.mean())

name=[n[:-9] for n in df_sbert.columns]

x=np.arange(4)
width=0.2
multiplier=0

plt.figure(figsize=(8,5),dpi=150)

for n, v in zip(["sbert","Augsbert"], [sbert,aug]):
    offset = width * multiplier
    m=plt.bar(x + offset, np.array(v), width, label=n,alpha=0.7,bottom=0)
    plt.bar_label(m,fmt='%.3f',padding=2,fontsize=6,color="#555555")
    multiplier += 1

plt.title('Spearman Correlation score with different measurements',y=1.02)
plt.ylabel('Spearman Correlation')
plt.xticks(x+0.1, name)


plt.legend(loc='upper left', ncols=2, framealpha=0.3, edgecolor="None",fontsize=9,labelcolor="#555555")
plt.ylim(top=1)

plt.show()