#!/usr/bin/env python
# coding: utf-8

# # <center><font color='crimson'><b>Sales Prediction in Retail</center>
# 
# ## <center><font color='green'><b>Domain : Retail</center>

# ### Objective :
# - Goal of this project is to **predict the sales** of a Retail outlet based on the historical data provided for sales.

# ### Feature Description :
# 
# **Item Identifier**: A code provided for the item of sale <br>
# 
# **Item Weight**: Weight of item <br>
# 
# **Item Fat Content**: A categorical column of how much fat is present in the item : ‘Low Fat’, ‘Regular’, ‘low fat’, ‘LF’, ‘reg’ <br>
# 
# **Item Visibility**: Numeric value for how visible the item is  <br> 
# 
# **Item Type**: What category does the item belong to: ‘Dairy’, ‘Soft Drinks’, ‘Meat’, ‘Fruits and Vegetables’, ‘Household’, ‘Baking Goods’, ‘Snack Foods’, ‘Frozen Foods’, ‘Breakfast’, ’Health and Hygiene’, ‘Hard Drinks’, ‘Canned’, ‘Breads’, ‘Starchy Foods’, ‘Others’, ‘Seafood’. <br>
# 
# **Item MRP**: The MRP price of item <br>
# 
# **Outlet Identifier**: Which outlet was the item sold. This will be categorical column <br>
# 
# **Outlet Establishment Year**: Which year was the outlet established <br>
# 
# **Outlet Size**: A categorical column to explain size of outlet: ‘Medium’, ‘High’, ‘Small’.  <br> 
# 
# **Outlet Location Type**: A categorical column to describe the location of the outlet: ‘Tier 1’, ‘Tier 2’, ‘Tier 3’  <br>
# 
# **Outlet Type** : Categorical column for type of outlet: ‘Supermarket Type1’, ‘Supermarket Type2’, ‘Supermarket Type3’, ‘Grocery Store’  <br>
# 
# **Item Outlet Sales**: The amount of sales for an item.  <br> 
# 
# **Source**: Whether the data is from train or test.  <br>

# ### Importing Required Modules

# - Firstly we'll load required modules and data to our notebook.
# 
# 
# - Later, we'll do data cleansing/pre-processing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('Train_Retail.csv', header=0)
train.head()


# In[ ]:


test = pd.read_csv('Test_Retail.csv', header=0)
test.head()


# In[ ]:


train.shape, test.shape


# - The Shape of Training set is 
#     - **Examples : 8532**
#     - **Features : 12 (including Target)**
#     
# - The Shape of Test set is
#     - **Examples : 5681**
#     - **Features : 11 (Target to Predict)**

# In[ ]:


train.info(), test.info()


# - Above result shows us the datatypes of our features
# 
# 
# - We got
#     - **(4) x float64**
#     - **(1) x int64**
#     - **(7) x object**

# ### EDA , Data Cleansing & Pre-processing

# In[ ]:


#splitting target variable from training set
#and merging train + test datasets for preprocessing

target = train['Item_Outlet_Sales']
train = train.drop(['Item_Outlet_Sales'], axis=1)

df = pd.concat([train, test])
df.head()


# - We're **splitting Target feature** from our **Train set.**
# 
# 
# - And We're **combining Train + Test Set** for **faster preprocessing / data cleansing.**
# 
# 
# - This approach is widely followed nowdays.

# In[ ]:


#Overall shape of merged dataset

df.shape


# - Now, overall shape of our dataset is
#     - **Examples : 14204**
#     - **Features : 11**

# In[ ]:


#Checking Null Values

df.isnull().sum()


# - We can observe that we have missing values in our dataset.
# 
# 
# - **Item_Weight** has **2439 Missing values** which comprises of **~17% of Missing Values** in that feature.
# 
# 
# - While, **Outlet_Size** has **4016 Missing values** which comprises of **~28% of Missing Values** in that particular feature.

# In[ ]:


#Descriptive Statistics

#sns.set_context('talk')
#plt.figure(figsize=(15,10))
#sns.heatmap(df.describe(), annot=True)
df.describe()


# **Descriptive Statistics for Whole Dataset**
# 
# - **Item_Weight** has values ranging from **4.55 to 21.35 with a mean value of 12.60** (adjusted to 2 decimal points). Count is showing as **11765** although we have **14204** number of rows in our dataset. This phenomenon is justified by missing value count of **2439**.
# 
# 
# - **Item_MRP** has values ranging from **31.29 to 266.88 with a mean value of 142.24**.
# 
# 
# - **Outlet_Establishment_Year** has values ranging from **1985 to 2009**. From this we can see that, first store established in 1985 and the most recent store begun in 2009.

# In[ ]:


#Unique code provided for item of sale

df['Item_Identifier'].nunique()


# - We got 1559 unique Item Code/Identifier.

# In[ ]:


#Lets see how item weight is distributed among item type

plt.figure(figsize=(15,8))
sns.set_context('talk')
sns.set_style('darkgrid')
sns.boxplot(df['Item_Type'], df['Item_Weight'], palette='cividis')
plt.title('Distribution of Item weight by its type')
plt.xticks(rotation=90)


# - We got **16 different Item Types** in our dataset which includes **Dairy, Soft Drinks, Meat, Snacks, Frozen food, etc**
# 
# 
# - We can observe that **different types of Items hold different weight.**
# 
# 
# - We can further **fill Null values in Item_Weight considering the Mean Weight of these Item types.**  

# In[ ]:


#Let's check Item Fat content

df['Item_Fat_Content'].value_counts()


# - Exploring the above results we can observe that we've **"Low Fat", "low fat" & "LF"** which is obviously nothing but the same.
# 
# 
# - Similarly, **"Regular" & "reg"** is not different.
# 
# 
# - We can **replace** them so we would get just 2 types of Fat Content , i.e, **"Regular" & "Low Fat".**

# In[ ]:


#replacing Fat type values to known values
#Plot Fat type count and plot how Fat content affects sales

plt.figure(figsize=(20,8))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.subplot(1,2,1)
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
sns.countplot(df['Item_Fat_Content'], palette='YlGn')
plt.title('Fat Content - Item Count')

plt.subplot(1,2,2)
sns.lineplot(df['Item_Fat_Content'][:train.shape[0]], target, color='g')
plt.title('Fat content Impact on Sales')


# - From plot (1) or L.H.S plot , we can observe that we've ample units of Low Fat content Items than that of Regular Fat content.
# 
# 
# - Plot (2) or R.H.S plot, gives us a brief idea of how Sales are being affected by just Fat content. **Low Fat content Items are sold quite less than Regular Fat content Items despite of having ample units of Low Fat content Items in Store.**
# 
# 
# - Let us also encode this ahead.

# In[ ]:


#encoding Fat content

df['Item_Fat_Content'].replace(('Low Fat', 'Regular'), (0, 1), inplace=True)


# In[ ]:


#Avg. weight of Item by Item type

avg_weight = df.groupby(['Item_Type'])['Item_Weight'].mean()
avg_weight = avg_weight.to_frame().reset_index()


# - Avg. weight is nothing but **Mean weight of Item** by **Item Type** so we can further **fill NaN values in Item_Weight.**

# In[ ]:


'''for i in np.arange(len(avg_weight)):
    x, y = avg_weight.iloc[i]
    print (x, y)'''


# In[ ]:


#Let's reset index of dataframe before proceeding into null value filling process.

df.reset_index(inplace=True)
df = df[df.columns[1:]]
df['new_item_weight'] = np.nan


# In[ ]:


#we'll fill null values in Item_weight with respect to Item Type

def item_weight(avg_weight_by_type):
    for i in np.arange(len(avg_weight_by_type)):
        x, y = avg_weight.iloc[i]
        sliced_df = df[df['Item_Type'] == str(x)]
        sliced_replaced_values = df['Item_Weight'][sliced_df.index].replace(np.nan, y)
        df['new_item_weight'][sliced_replaced_values.index] = sliced_replaced_values.values
        


# - We had created **User-Defined function** to fill NaN's in sliced dataframe by its Item_type.
# 
# 
# - We are creating new_item_weight feature which will have all the values of Item_Weight but NaN's will be filled.
# 
# 
# - We can further drop Item_Weight.
# 
# 
# - Let's intialize our defined function ahead.

# In[ ]:


#Applying user defined function

print (item_weight(avg_weight))


# In[ ]:


#Lets see some samples of data to see if our new column has similar values of Item_weight with replaced NaN

df[['Item_Weight', 'new_item_weight']].sample(10)


# - Above is sample of **Item_Weight** & **new_item_weight** at same index, we can observe that the row information at index 12013 has NaN in Item_Weight but new_item_weight has the NaN fixed.
# 
# 
# - Which concludes that we've filled **Null Values** with mean weight of that **Item_Type.**
# 
# 
# - Let us also drop our **Item_Weight** as we have **new_item_weight**

# In[ ]:


df.drop(['Item_Weight'], axis=1, inplace=True)


# In[ ]:


#Numeric value for how visible the item is

sns.set_context('talk')
sns.jointplot(df['Item_Visibility'][:train.shape[0]], target, kind='kde', color='crimson', height=8, )


# - Above is the beautiful vizualization which depits **Item Sales by Item Visibility**. In short, how item is being sold by how much visible it is in the store racks.
# 
# 
# - **0.0 meaning it's in front, meaning more visible while 0.1 is less visible** and so on.
# 
# 
# - And we can observe that **more visible Items are sold more, less visible are sold less.**

# In[ ]:


#Lets mean encode Item visibility for Vizualization

def mean_encode(data):
    if data <= 0.1:
        return 1
    if data <= 0.2 and data > 0.1:
        return 2
    if data <= 0.3 and data > 0.2:
        return 3
    if data <= 0.4 and data > 0.3:
        return 4
    
df['Item_Visibility_mean_enc'] = df['Item_Visibility'].apply(lambda x : mean_encode(x))


# **Alternate Way**
# 
# - As we have Item_visibility in continuous form we can mean encode it by user_defined function , i.e, less than equals 0.1 can be considered as 1, less than equals 0.2 but greater than 0.1 can be encoded as 2 and so on. 
# 
# 
# - This approach will give us categorical values , i.e, 1,2,3,4 in our case stating 1 as more visible Items, 2 is less visible , etc.
# 
# 
# - And we can observe it in below plot how the vizibility impacts sales.

# In[ ]:


plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.countplot(df['Item_Visibility_mean_enc'], palette='rainbow')
plt.title('Mean Encoded Visibility')

plt.subplot(1,2,2)
sns.boxenplot(df['Item_Visibility_mean_enc'][:train.shape[0]], target, palette='rocket')
plt.ylabel('Sales')
plt.xlabel('Mean Encoded - Visibility')
plt.title('Imapact of Item Visibility on Sales')


# - Above R.H.S plot is the alternate plot depicting us the same like we saw earlier that visible items were sold more and less visible didn't do much.
# 
# 
# - Item falling in 1 & 2 category of visibility were sold more.

# In[ ]:


#Let's look at MRP as per Item type

plt.figure(figsize=(18,8))
sns.stripplot(df['Item_Type'], df['Item_MRP'], palette='rainbow', dodge=True, alpha=0.5, zorder=1)
plt.xticks(rotation=90)


# - Above Item shows us how Maximum Retail Price of Item per Item type.
# 
# 
# - Ofcourse , we have different values for different types.
# 
# 
# - But this vizualization shows us the distribution of MRP by Item type. 
# 
# 
# - We can observe that breakfast & seafood item is ranging from ~20 to 230 MRP while others Item types are ~20 to 270 MRP.

# In[ ]:


#Let's encode Item Type

items = df['Item_Type'].value_counts().index

encoding_values = range(len(items))

df['Item_Type'].replace(items, encoding_values, inplace=True)


# - We have encoded Item_Type above.

# In[ ]:


#Outlet Identifier

plt.figure(figsize=(20,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Identifier'], palette='prism')
plt.xticks(rotation=90)
plt.title('Outlets')

plt.subplot(1,2,2)
sns.boxenplot(df['Outlet_Identifier'][:train.shape[0]], target, palette='prism')
plt.xticks(rotation=90)
plt.title('Impact of outlets on Sales')


# - We can observe that **Out027** had made good amount of Sales compared to other outlets.
# 
# 
# - While, **Outlet10 , Outlet19** did least amount of sales among others.

# In[ ]:


plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Establishment_Year'], palette='inferno')
plt.xticks(rotation=90)
plt.title('Outlet Establishment Year')

plt.subplot(1,2,2)
sns.boxplot(df['Outlet_Establishment_Year'][:train.shape[0]], target, palette='inferno')
plt.xticks(rotation=90)
plt.title('Impact on Sales due to Outlet Establishment')


# - We could observe that when store first established in **1985** the sales were **booming** , maybe because of **curosity of people to try our store.** But we need to also consider **location / regional factor.** Like we see in the case of 1998's outlets.
# 
# 
# - There is clear **dip** in Sales which is identified from Outlet opened in **1998.**
# 
# 
# - **What can be the reason ?** Ofcourse, could be the **regional / location factor.**

# - Let us also encode **Outlet_Identifier, Outlet_Size, Outlet_Location_Type** & **Outlet_Establishment_Year.**

# In[ ]:


#Let's encode Outlet Identifier

outlets = df['Outlet_Identifier'].value_counts().index

enc_outlets = range(len(outlets))

df['Outlet_Identifier'].replace(outlets, enc_outlets, inplace=True)


# In[ ]:


#Let's encode Outlet Establishment year

years = df['Outlet_Establishment_Year'].value_counts().index.sort_values()

enc_years = range(len(years))

df['Outlet_Establishment_Year'].replace(years, enc_years, inplace=True)


# In[ ]:


#Encoding Outlet Size
#High, Tier 3 = 3
#Medium, Tier 2 = 2
#Small, Tier 1 = 1

df['Outlet_Size'].replace(('High', 'Medium', 'Small'), (3,2,1), inplace=True)
df['Outlet_Location_Type'].replace(('Tier 3', 'Tier 2', 'Tier 1'), (3,2,1), inplace=True)


# In[ ]:


#Outlet Size

plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Size'], palette='cool')
plt.title('Outlet Size')

plt.subplot(1,2,2)
sns.boxenplot(df['Outlet_Size'][:train.shape[0]], target, hue=df['Outlet_Location_Type'][:train.shape[0]], palette='brg')
plt.xticks(ticks=(0, 1, 2), labels=('Small', 'Medium', 'High'))
label = ['Tier 1', 'Tier 2', 'Tier 3']
plt.legend(loc='upper left')


# - From R.H.S. plot we can conclude that :
# 
#     - **Small Outlets** are present in **Tier 1** and **Tier 2** locations.
#     - **Medium Outlets** are present in **Tier 1** and **Tier 3** locations.
#     - **Big / High outlets** are present only in **Tier 3** Locations.
#     - In **Tier 3** city the **High Outlet & Medium Outlet** are impacting more on **Sales.**

# In[ ]:


#Outlet type Analysis

plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Type'], palette='icefire_r')
plt.title('Outlet Type')
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.boxenplot(df['Outlet_Type'][:train.shape[0]], target, hue=df['Outlet_Location_Type'][:train.shape[0]], palette='icefire_r')
plt.title('Outlet Type Impact on Sales')
plt.xticks(rotation=90)
plt.legend(loc='upper left')


# - In R.H.S. Plot, we can observe that **Supermarket 1** is present in **All 3 Tiers Locations.**
# 
# 
# - **Supermarket 2 and 3** is only found in **Tier 3 location.**
# 
# 
# - While, **Grocery Store** is found in **Tier 1 and Tier 3 locations.**
# 
# 
# - Let us also encode **Outlet_Type.** 

# In[ ]:


#Encoding Outlet Size

df['Outlet_Type'].replace(('Supermarket Type3', 'Supermarket Type2', 'Supermarket Type1', 'Grocery Store'), (3,2,1,0), inplace=True)


# In[ ]:


#Fetching Item Id's to known form 

df['Item_Identifier'] = df['Item_Identifier'].apply(lambda x : x[:2])


# - Let us fetch **Prefix of Item ID**, because we see that there are **identical** for every other **Item_Type** that we've in our dataset.

# In[ ]:


sns.countplot(df['Item_Identifier'], palette='spring_r')
plt.title('Item IDs')


# - **FD's** are **more dominant** in our dataset.
# 
# 
# - Encoding **Item ID's** ahead.

# In[ ]:


#Label Encoding Item IDs

df['Item_Identifier'].replace(('FD', 'DR', 'NC'), (1,2,3), inplace=True)


# In[ ]:


df.head()


# In[ ]:


#Let's fill NaN values now using FancyInput's IterativeImputer

from fancyimpute import IterativeImputer

df_filled = pd.DataFrame(data=IterativeImputer(imputation_order='roman', n_iter=50, initial_strategy='mean').fit_transform(df.values), columns=df.columns, index=df.index)
df_filled.head()


# - Above we're using **Fancy Impute's "Iterative Imputer"** to fill NaN values of **Outlet_Size.**
# 
# 
# - We'll Iterate it for **50 times** having initial strategy as **mean / average.**

# In[ ]:


#Checking null values

df_filled.isnull().any()


# - We can see that all Null values are replaced as we needed.

# In[ ]:


#Lets see again how item weight is distributed among item type

plt.figure(figsize=(15,8))
sns.set_context('talk')
sns.boxplot(df_filled['Item_Type'], df_filled['new_item_weight'], palette='cividis')
plt.title('Distribution of Item weight by its type')
plt.xticks(rotation=90)


# - Let's us revist the same vizualization after filling Null values.
# 
# 
# - We can see there is a slight change / movement in weights **median** values since we've replaced those NaN's with **Average Item_Weight** by **Item_Type.**

# In[ ]:


'''plt.figure(figsize=(20,15))
sns.heatmap(df_filled.corr()*100, square=True, annot=True, cmap='Wistia_r')
plt.title('Pearson Correlation')'''


# - By examining the above Correlation plot (x100 - For better understanding), we can say that there is no correlation between variables except **Mean_Item_Visibility** and **Item_visibility**
# - We can drop one of the column.

# In[ ]:


#Let's drop Item_visibility, Item_MRP, new_item_weight and Item_visibility_mean_enc because we will take Log of those.

df_filled.drop(['Item_Visibility', 'Item_Visibility_mean_enc',
                'new_item_weight', 'Item_MRP'
                ], axis=1, inplace=True)


# - Let's **drop** the above features as we going to **Log. transform** them to **reduce the effect of Outliers** on our statistical model.

# In[ ]:


#Distribution of Item Weight, Item MRP & Item Visibility

plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,3,1)
sns.distplot(df['new_item_weight'], color='black')
plt.title('Item Weight Distribution')

plt.subplot(1,3,2)
sns.distplot(df['Item_MRP'], color='r')
plt.title('Item MRP Distribution')

plt.subplot(1,3,3)
sns.distplot(df['Item_Visibility'], color='g')
plt.title('Item Visibility Distribution')


# - Above plot depicts the **distribution** of **new_item_weight, Item_MRP & Item_Visibility.**
# 
# 
# - We're going to **Log Transform** all of these to reduce the effects of Outliers.

# In[ ]:


#Log Transformation

plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,3,1)
log_item_weight = np.log1p(df['new_item_weight'])
sns.distplot(log_item_weight, color='black')
plt.title('Log. Item Weight Distribution')

plt.subplot(1,3,2)
log_item_mrp = np.log1p(df['Item_MRP'])
sns.distplot(log_item_mrp, color='r')
plt.title('Log. Item MRP Distribution')

plt.subplot(1,3,3)
log_item_vis = np.log1p(df['Item_Visibility'])
sns.distplot(log_item_vis, color='g')
plt.title('Log. Item Visibility Distribution')


# - Above plot is after **Log Transformation** of those 3 features.

# In[ ]:


df_filled['log_item_weight'] = log_item_weight
df_filled['log_item_mrp'] = log_item_mrp
df_filled['log_item_visibility'] = log_item_vis

df_filled.head()


# - Let us assign those **Log. Transformed** values to our **df_filled dataframe.**

# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(df_filled.corr()*100, square=True, annot=True, cmap='Wistia_r')
plt.title('Pearson Correlation')


# - By examining the above **Correlation plot** (x100 - For better understanding), we can say that there is **no strong positive correlation between variables.**
# 
# 
# - Let us **Normalize** the dataset using **Sklearn's Standard Scaler.**

# In[ ]:


#Let's scale our values

from sklearn.preprocessing import StandardScaler

cols = df_filled.columns

scaler = StandardScaler()

norm_df = scaler.fit_transform(df_filled.values)
norm_df = pd.DataFrame(data=norm_df, columns=cols, index=df_filled.index)
norm_df.head()


# - Further let us **split** out dataset and bring it to Train / Test shape.
# 
# 
# - Also, Let us observe the distribution of target variable and **Log. Transform** it.

# In[ ]:


#Let's split train and test data

train = norm_df[:train.shape[0]]
test = norm_df[train.shape[0]:]

print ('Shape of Training Set : ',train.shape)
print ('Shape of Testing Set : ',test.shape)


# In[ ]:


#Lets look at Target variable 

plt.figure(figsize=(22,7))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.subplot(1,2,1)
sns.distplot(target, color='indigo')
plt.title('Item Outlet Sales')

plt.subplot(1,2,2)
log_target = np.log1p(target)
sns.distplot(log_target, color='indigo')
plt.title('Log. Item Outlet Sales')


# In[ ]:


#Lets form training and validation data

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_log_error , mean_squared_error

x = train
y = log_target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ### Null RMSLE

# - Null RMSLE is the RMSLE that could be achieved by predicting the mean response value. It is a benchmark against which you may want to measure your regression model.

# In[ ]:


y_null = np.zeros_like(y_test)
y_null.fill(y_test.mean())
print ("Null RMSLE : ", np.sqrt(mean_squared_log_error(y_test, y_null)))


# ### Building a Statistical Model
# 
# - In this step we are going to create data models that are capable of predicting the Outlet_Sales. Inorder to create these models first had already split the given Train dataset into Train and Validation datasets. Train dataset is the one which have the Logarithm of Item_Outlet_Sales values. We will train the model based on these values to validate the Validation dataset values.
# 
# 
# 
# - First we are going to create a basic **Linear Regression** model.

# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

est_lr = LinearRegression()
est_lr.fit(x_train, y_train)
lr_pred = est_lr.predict(x_test)
lr_pred


# In[ ]:


est_lr.intercept_, est_lr.coef_


# **Intuition from Co-efficients :**
# 
# - We've these features :
# 
# **Item_Identifier, Item_Fat_Content, Item_Type, Outlet_Identifier,**<br>
# **Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type,**<br>
# **log_item_weight, log_item_mrp, log_item_visibility**
# 
# 
# - And we got these Co-efficients :
# 
# 
# **0.00244514, 0.01002893, 0.00790689, -0.48508059,**<br>
# **0.35319793, 0.13828741, -0.3117243, 0.3420772 ,**<br>
# **0.00107887, 0.53274457, -0.02143053**
# 
# 
# - We can say that each item sale largely depends on **MRP, Outlet Establishment year** & **Outlet Type.** 
# 
# 
# - It is Surprising, **Outlet_Establishment_Year** is also one prominent feature affecting sales.
# 
# 
# 
# - <b>Note : We should clearly keep in mind that all these variables are inter-connected. We can't really identify how much an individual variable effects the Item_Outlet_Sales. Instead we assume that by keeping all the remaining variables fixed, how much a unit change in one variable effects overall Item_Outlet_Sales.

# In[ ]:


print ("Training Score : " , est_lr.score(x_train, y_train))

print ("Validation Score : ", est_lr.score(x_test, y_test))

print ("Cross Validation Score : " , cross_val_score(est_lr, x_train, y_train, cv=5).mean())

print ("R2_Score : ", r2_score(lr_pred, y_test))

print ("RMSLE : ", np.sqrt(mean_squared_log_error(lr_pred, y_test)))


# - We got **Training Score, Validation Score & Croos-Validation Score** approximately between **0.69 - 0.70.**
# 
# 
# - Our evaluation metric is **RMSLE (Root-Mean Squared Logarithmic Error)**
# 
# 
# - The Lesser the value of RMSLE (than Null RMSLE), we can say our model is a good fit. 
# 
# 
# - **Linear Regression model** is giving us **RMSLE of 0.072** while our **Null RMSLE is 0.133** which means **0.072 < 0.133** concludes our model is a good fit.
# 
# 
# - Although we can try to **reduce RMSLE more by using other Machine Learning algorithms.**

# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(lr_pred, kde=True, color="orange", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")


# - Vizualizations are always better way of representing the data.
# 
# 
# - The above viz. shows us **how close our predicted sales are to actual sales.** 
# 
# 
# - We can say that **Linear Regression** is acceptable model.

# #Decision Tree

# In[3]:


'''from sklearn.tree import DecisionTreeRegressor

est_dt = DecisionTreeRegressor(criterion="mse", max_depth=5)
est_dt.fit(x_train, y_train)
dt_pred = est_dt.predict(x_test)
dt_pred'''


# In[4]:


'''print ("Training Score : " , est_dt.score(x_train, y_train))

print ("Validation Score : ", est_dt.score(x_test, y_test))

print ("Cross Validation Score : " , cross_val_score(est_dt, x_train, y_train, cv=5).mean())

print ("R2_Score : ", r2_score(dt_pred, y_test))

print ("RMSLE : ", np.sqrt(mean_squared_log_error(dt_pred, y_test)))'''


# In[5]:


'''plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(dt_pred, kde=True, color="olive", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")'''


# #Random Forest

# In[6]:


'''from sklearn.ensemble import RandomForestRegressor

est_rf = RandomForestRegressor(criterion="mse", n_estimators=10, max_depth=5)
est_rf.fit(x_train, y_train)
rf_pred = est_rf.predict(x_test)
rf_pred'''


# In[7]:


'''print ("Training Score : " , est_rf.score(x_train, y_train))

print ("Validation Score : ", est_rf.score(x_test, y_test))

print ("Cross Validation Score : " , cross_val_score(est_rf, x_train, y_train, cv=5).mean())

print ("R2_Score : ", r2_score(rf_pred, y_test))

print ("RMSLE : ", np.sqrt(mean_squared_log_error(rf_pred, y_test)))'''


# In[8]:


'''plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(rf_pred, kde=True, color="mediumblue", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")'''


# ### CatBoost Regressor

# - We'll install one of the most popular module / algorithm , i.e, **CatBoost.**

# In[ ]:


#!pip install catboost


# - We'll initialize CatBoost for **1000 Iterations** and **evaluate RMSLE** we get from it.
# 
# 
# - And further compare **RMSLE** with **Linear Regression's RMSLE** and **Null RMSLE.**

# In[ ]:


from catboost import CatBoostRegressor

est_cbr = CatBoostRegressor(iterations=1000, eval_metric='RMSE')
est_cbr.fit(x_train, y_train)
cbr_pred = est_cbr.predict(x_test)
cbr_pred


# In[ ]:


#print ("Training Score : " , est_cbr.score(x_train, y_train))

#print ("Validation Score : ", est_cbr.score(x_test, y_test))

#print ("Cross Validation Score : " , cross_val_score(est_cbr, x_train, y_train, cv=5).mean())

print ("R2_Score : ", r2_score(cbr_pred, y_test))

print ("RMSLE : ", np.sqrt(mean_squared_log_error(cbr_pred, y_test)))


# - Fantastic !!! We are getting best RMSLE so far.
# 
# 
# - **Linear Regression gave us 0.072** while **CatBoost is giving us 0.066** which is far lesser than our Null RMSLE's benchmark of **0.133.**
# 
# 
# - Let us also look below through viz. how close we're to actual values.

# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(cbr_pred, kde=True, color="deeppink", label="Prediction")
plt.legend()
plt.title("Test VS Prediction")


# - We can further build a XGB Regressor to figure out its performance as well.

# ### XGB

# In[ ]:


from xgboost import XGBRegressor

est_xgb = XGBRegressor(max_depth=3, learning_rate=0.06, n_estimators=91, importance_type='total_gain'
                        )
est_xgb.fit(x_train, y_train)

xgb_pred =  est_xgb.predict(x_test)
#print ('Prediction : ', pred, '\n')

print ('Training Score : ', est_xgb.score(x_train, y_train))
print ('Validation Score : ', est_xgb.score(x_test, y_test))
print ('Cross Validation Score : ', cross_val_score(est_xgb, x_train, y_train, cv=5).mean())
#print ("R2_Score : ", r2_score(xgb_pred, y_test))
print ("RMSLE : ", np.sqrt(mean_squared_log_error(xgb_pred, y_test)))


# - We can observe that XGB's **training, validation and Cross-Validation scores** between **0.73 to 0.75** which is acceptable. As 2% to 3% of difference between scores is said to be acceptable.
# 
# 
# 
# - The **RMSLE** which we got here is similar to **CatBoost's RMSLE**. But even slight difference can make model better. Let us evaluate all models to getter better idea.

# In[ ]:


sns.set_context('talk')
plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(xgb_pred, kde=True, color="limegreen", label="Prediction")
plt.legend()
plt.title("XGB - Test VS Prediction")


# - Viz. above depicts XGB's sales prediction closeness to actual sales.

# #LGBM

# In[11]:


'''from lightgbm import LGBMRegressor

est_lgbm = LGBMRegressor(boosting_type='gbdt', max_depth=3, learning_rate=0.09, importance_type='total_gain',
                        )
est_lgbm.fit(x_train, y_train)

lgbm_pred =  est_lgbm.predict(x_test)
#print ('Prediction : ', pred, '\n')

print ('Training Score : ', est_lgbm.score(x_train, y_train))
print ('Validation Score : ', est_lgbm.score(x_test, y_test))
print ('Cross Validation Score : ', cross_val_score(est_lgbm, x_train, y_train, cv=5).mean())
#print ("R2_Score : ", r2_score(lgbm_pred, y_test))
print ("RMSLE : ", np.sqrt(mean_squared_log_error(lgbm_pred, y_test)))'''


# In[12]:


'''plt.figure(figsize=(15,8))
plt.subplot(1,1,1)
sns.distplot(y_test, kde=True, color="black", label="Test")

plt.subplot(1,1,1)
sns.distplot(lgbm_pred, kde=True, color="chartreuse", label="Prediction")
plt.legend()
plt.title("LGBM - Test VS Prediction")'''


# ### Model Evaluation :
# 
# 
# - In this phase of the project we're going to evaluate the models that we've built so far. We will compare the RMSLE obtained from different models.<br><br>
# 
#     - <b>RMSLE for ***Linear Regression*** - 0.0725
#     - <b>RMSLE for ***CatBoost Regressor*** - 0.0665
#     - <b>RMSLE for ***XGB Regressor*** - 0.667
# 
# 
# - **CatBoost with 1000 Iterations** had performed the best and we will use it to make predictions from the test data.

# ### Prediction on Test Set

# - Further we'll make predictions on test data using **CatBoost.**
# 
# 
# - Since we had **Log. Transformed** our target while doing Pre-processing we will need to convert those values in **Exponential form.** For that we'll use **expm1** as we used **log1p for Log. Transformations.**
# 
# 
# - And at last we'll read **Item_Itendifier** and create new dataframe and glue our **Predictions** to it and write it to **csv.**

# In[ ]:


test_id = pd.read_csv('Test_Retail.csv', header=0)['Item_Identifier']

final_pred = np.expm1(est_cbr.predict(test))
final_pred = pd.DataFrame({'Item_Identifier':test_id.values, 'Item_Outlet_Sales':final_pred})
final_pred.set_index('Item_Identifier', inplace=True)
final_sub = final_pred.to_csv('Final_Sub.csv')


# - Below is the Viz. for **Distribution of Exp. Predictions**

# In[100]:


plt.figure(figsize=(15,8))
sns.distplot(np.expm1(est_cbr.predict(test)))
plt.title('Exp. Prediction')


# ### Conclusion :
# 
# - We built a acceptable statistical model that predicts Sales of Retail Store. After few experimentations with various ML algorithms , **CatBoost model** turned out to be best for us.
# 
# 
# ### What's next ?
# 
# - We can improve further by doing **Hyper-Parameter** tuning using **GridSearchCV.**
# 
# - One can Try **Deep Neural Network** Implementation.
# 
# - One can also try **AutoML** approach.
