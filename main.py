import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize ,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
creditcard_df= pd.read_csv('../input/creditcard/Credit_card_data.csv')
creditcard_df
print(creditcard_df.isnull().sum())


# In[10]:


creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()          
creditcard_df.isnull().sum()


# In[8]:


creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True),'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()


# In[11]:


creditcard_df.isnull().sum()


# In[12]:


creditcard_df.isnull().sum().sum()


# In[13]:


creditcard_df.drop('CUST_ID',axis=1,inplace=True)


# In[14]:


creditcard_df


# In[15]:


n = len(creditcard_df.columns)
n


# In[16]:


creditcard_df.columns


# In[19]:


correlations=creditcard_df.corr()
sns.heatmap(correlations,annot=True)


# In[20]:


scaler=StandardScaler()
creditcard_df_scaled=scaler.fit_transform(creditcard_df)


# In[21]:


type(creditcard_df_scaled)


# In[22]:


creditcard_df_scaled


# In[23]:


cost=[]
range_values=range(1,20)
for i in range_values:
    kmeans=KMeans(i)
    kmeans.fit(creditcard_df_scaled)
    cost.append(kmeans.inertia_)
plt.plot(cost)


# In[24]:


kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)#find the nearest clusters for given data
labels = kmeans.labels_
labels


# In[25]:


kmeans.cluster_centers_.shape


# In[26]:


cluster_centers=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])
cluster_centers


# In[27]:





# In[28]:


labels.shape


# In[29]:


labels.max()


# In[30]:


labels.min()


# In[31]:


credit_df_cluster=pd.concat([creditcard_df,pd.DataFrame(({'cluster':labels}))],axis=1)
credit_df_cluster


# In[32]:


pca=PCA(n_components=2)
principal_comp=pca.fit_transform(creditcard_df_scaled)


# In[33]:


pca_df=pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])
pca_df


# In[34]:


pca_df=pd.concat([pca_df,pd.DataFrame({'Cluster':labels})],axis=1)
pca_df


# In[36]:


plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue='Cluster',data=pca_df,palette=['yellow','red','blue','pink','orange','black','purple'])
plt.show()
