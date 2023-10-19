import pandas as pd

# df=pd.read_csv("raw data/stsbenchmark.tsv",delimiter='\t',header=0,quoting=3)
# df=df.drop(df.columns[1:5],axis=1)


# df.to_csv("stsbenchmark.tsv",sep='\t',index=False)


# df1=pd.read_csv("raw data/msr_paraphrase_train.txt",delimiter='\t',quoting=3)
# df2=pd.read_csv("raw data/msr_paraphrase_test.txt",delimiter='\t',quoting=3)
# df3=pd.read_csv("raw data/dev_ids.tsv",delimiter='\t',names=["#1 ID","#2 ID"])



# df=pd.concat([df1,df2],ignore_index=True)
# df.insert(0,"split",len(df1)*["train"]+len(df2)*["test"])
# df.drop(df.columns[[2,3]],axis=1,inplace=True)
# df.columns=['split','score','sentence1','sentence2']

# dfids=df1[["#1 ID","#2 ID"]]

# for i,r in df3.iterrows():
#     ind=dfids[dfids==r].dropna().index
#     df.iloc[ind,0]=['dev']

# df=df.to_csv("msr_paraphrase.tsv",sep='\t',index=False)

df=pd.read_csv("mrpc/msr_paraphrase.tsv",delimiter='\t',header=0,quoting=3)
print((df["split"]=="train").sum())
print(len(df))