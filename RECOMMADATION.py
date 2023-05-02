#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 100)
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


df=pd.read_csv('recom.csv')


# In[3]:


df.head()


# In[4]:


df['Transaction_ID'].nunique()


# In[5]:


df = pd.read_csv('recom.csv', usecols=lambda column: column not in ['Unnamed: 0', 'ItemKey'])


# In[6]:


df.info()


# In[7]:


#Toplam harcamayi Column olarak ekliyoruz
df['TotalPrice']=df['Amount']*df['Price']


# In[8]:


#siparis tarihinin veri tipini degistiriyoruz
df['Date']=pd.to_datetime(df['Date'])


# In[9]:


# bugunu/ analiz tarihini degisken olarak atiyoruz.
today= dt.datetime(2023,1,1)
print(today)


# In[10]:


df.describe().T


# In[11]:


# minimumu 0 olan deger yok.. silmeye gerek yok!


# In[12]:


# Recency ve Monetary degerlerini bulalim
df_x=df.groupby('Main_ID').agg({'TotalPrice': lambda x: x.sum(),'Date':lambda x: (today-x.max()).days})


# In[13]:


#Transaction_ID code halinde oldugu icin Transaction_ID'yi (Invoice) unique hale getiriyoruz.
df_y=df.groupby(['Main_ID','Transaction_ID']).agg({'TotalPrice': lambda x: x.sum()})


# In[14]:


# saydirdigimizda Transaction_ID unique halde gelmis oluyor.
df_z=df.groupby('Main_ID').agg({'TotalPrice': lambda x: len(x)})


# In[15]:


#RFM tablosuna ulasmis oluyoruz
rfm_table = pd.merge(df_x, df_z, on='Main_ID')


# In[16]:


#column isimlerini belirliyoruz
rfm_table.rename(columns={'Date': 'Recency',
                          'TotalPrice_x':'Monetary',
                          'TotalPrice_y':'Frequency'},inplace=True)


# In[17]:


rfm_table.head()


# In[18]:


#Frequency bulma
def FScore(x,p,d):
    if x<=d[p][0.20]:
        return 0
    elif x<=d[p][0.40]:
        return 1
    elif x<=d[p][0.60]:
        return 2
    elif x<=d[p][0.80]:
        return 3
    else:
        return 4
    
    
quantiles= rfm_table.quantile(q=[0.20, 0.40, 0.60, 0.80])
quantiles=quantiles.to_dict()
rfm_table['Freq_Tile']=rfm_table['Frequency'].apply(FScore, args=('Frequency',quantiles))
#Recency bulma
rfm_table= rfm_table.sort_values('Recency', ascending=True)
rfm_table['Rec_Tile']=pd.qcut(rfm_table['Recency'],5,labels=False)

#Monetary bulma
# rfm_table= rfm_table.sort_values('Monetary', ascending=True)
rfm_table['Mone_Tile']=pd.qcut(rfm_table['Monetary'],5,labels=False)

# '0' degeri yer almasin istiyorsak, buldugumuz degerleri 1 artiririz
rfm_table['Rec_Tile']=rfm_table['Rec_Tile'] +1
rfm_table['Freq_Tile']=rfm_table['Freq_Tile']+1
rfm_table['Mone_Tile']=rfm_table['Mone_Tile']+1

#buldugumuz degerleri birlestirip tek 1 skor elde ediyoruz.
rfm_table['RFM Score']=rfm_table['Rec_Tile'].map(str)+rfm_table['Freq_Tile'].map(str)+rfm_table['Mone_Tile'].map(str)

rfm_table.head()



    


# In[19]:


#degerlerin iceriklerini inceliyoruz
rfm_table.groupby('RFM Score').agg({
    'Recency': ['mean','min','max','count'],
    'Frequency':['mean','min','max','count'],
    'Monetary':['mean','min','max','count'] }).round(1).head(20)


# In[20]:


rfm_table.groupby('RFM Score').size().sort_values(ascending=False)[:20]


# In[21]:


plt.figure(figsize=(6,6))
sns.countplot(x='Freq_Tile', data=rfm_table)
plt.ylabel('Count',fontsize=12)
plt.xlabel('Freq_Flag',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Frequency of Freq_Flag', fontsize=15)
plt.show()





# In[22]:


sns.distplot(rfm_table['Recency'])
plt.show()   

  


# In[23]:


sns.distplot(rfm_table[rfm_table.Frequency<20]['Frequency'])
plt.show()   


# In[24]:


sns.distplot(rfm_table['Monetary'])
plt.show()   


# In[25]:


rfm_table.head()


# In[26]:


clus= rfm_table[['Monetary','Recency','Frequency']]


# In[27]:


clus.head()


# In[28]:


clusterdata=clus.iloc[:,0:4]
clusterdata.head()


# In[29]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler= MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(clus)
data_scaled2= pd.DataFrame(x_scaled)
# Virgülden sonra 4 basamak gösterme ayarı
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))


# In[30]:


data_scaled2.head()


# In[31]:


# Virgülden sonra 2 basamak gösterme ayarı
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))


# In[32]:


data_scaled2.head()


# In[33]:


data_scaled2.describe().T


# In[34]:


get_ipython().run_line_magic('pinfo', 'KMeans')


# In[35]:


plt.figure(figsize=(8,6))
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++',n_init=10, max_iter=300)
    kmeans.fit(data_scaled2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()


# In[36]:


from sklearn.metrics import silhouette_score
inertia_list=[]
silhouette_score_list=[]
for i in range (2,10):
    #kmeans=KMeans(n_clusters=i, init='k-means++',n_init=10, max_iter=300)
    kmeans.fit(data_scaled2)
    silhouette_score_list.append(silhouette_score(data_scaled2,kmeans.labels_))
    print(silhouette_score_list)
    




# In[37]:


np.argmax(silhouette_score_list)+2


# In[38]:


#kmeans using 4 clusters and k-means++ initialization
kmeans= KMeans(n_clusters=4, init='k-means++',n_init=10, max_iter=300)
kmeans.fit(data_scaled2)
pred=kmeans.predict(data_scaled2)


# In[39]:


d_frame=pd.DataFrame(clus)
d_frame['cluster']=pred
d_frame['cluster'].value_counts()


# In[40]:


d_frame.head()


# In[41]:


d_frame.groupby('cluster').mean()


# ## Association Rules - Birliktelik Analizi

# In[42]:


get_ipython().system('pip install mlxtend')


# In[43]:


get_ipython().system('pip install mlxtend')
get_ipython().system('pip install --upgrade mlxtend')
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[49]:


df_apriori = df.groupby(['Transaction_ID', 'Code_Product'])['Amount'].sum().unstack().reset_index().fillna(0).set_index('Transaction_ID')


# In[50]:


df_apriori.head()


# In[55]:


def num(x):
    if x<=0:
        return 0
    if x >=1:
        return 1
sepet= df_apriori.applymap(num)


# In[56]:


sepet.head(10)


# In[57]:


from mlxtend.frequent_patterns import fpgrowth


# In[91]:


rule_fp= fpgrowth(sepet, min_support=0.0001, use_colnames=True)
rule_fp['support'] = rule_fp['support'].map('{:.4f}'.format)

rule_fp


# In[89]:


items= apriori(sepet, min_support=0.0001, use_colnames=True)


# In[90]:


rule = association_rules(items, metric="confidence", min_threshold=0.001)

print(rule)


# In[ ]:




