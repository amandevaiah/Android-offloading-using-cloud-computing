
import os
os.chdir("F:\Model Deployment\prediction_Expense")


import pandas as pd
MyData = pd.read_csv("Income_Expense_Data.csv")

MyData.shape


MyData.head(10)




MyData.isnull().sum() 




MyData["Income"].fillna((MyData["Income"].median()), inplace = True)




MyData.isnull().sum() 


# In[46]:


#Checking for outliers
MyData.describe()  #notice the maximum value in Age


# In[47]:


#Checking different percentiles
pd.DataFrame(MyData['Age']).describe(percentiles=(1,0.99,0.9,0.75,0.5,0.3,0.1,0.01))


# In[48]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(MyData['Age'])
plt.show()


# In[49]:



#getting median Age
Age_col_df = pd.DataFrame(MyData['Age'])
Age_median = Age_col_df.median()

#getting IQR of Age column
Q3 = Age_col_df.quantile(q=0.75)
Q1 = Age_col_df.quantile(q=0.25)
IQR = Q3-Q1

#Deriving boundaries of Outliers
IQR_LL = int(Q1 - 1.5*IQR)
IQR_UL = int(Q3 + 1.5*IQR)

MyData.loc[MyData['Age']>IQR_UL , 'Age'] = int(Age_col_df.quantile(q=0.99))
MyData.loc[MyData['Age']<IQR_LL , 'Age'] = int(Age_col_df.quantile(q=0.01))


# In[50]:


max(MyData['Age'])


x = MyData["Income"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Expense")


x = MyData["Age"]
y=  MyData["Expense"]


plt.scatter(x, y, label="Income Age")





correlation_matrix= MyData.corr().round(2)
f, ax = plt.subplots(figsize =(8, 4)) 
import seaborn as sns
sns.heatmap(data=correlation_matrix, annot=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(MyData)
scaled_data



MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ["Age","Income","Expense"]


features = ["Income","Age"]
response = ["Expense"]
X=MyData_scaled[features]
y=MyData_scaled[response]




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn import metrics


model = LinearRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test,y_test)
print(accuracy*100,'%')




model.predict(X_test)


import pickle
pickle.dump(model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30000, 24]]))






