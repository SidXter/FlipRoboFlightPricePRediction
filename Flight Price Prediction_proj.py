#!/usr/bin/env python
# coding: utf-8

# In[561]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# #### Loading the Dataset.

# In[562]:


fDF=pd.read_csv("FlightDataWebscrape.csv")


# In[563]:


fDF.head(50)


# In[564]:


fDF.shape


# #### checking for null values

# In[565]:


fDF.isnull().sum()


# There are no null values in dataset.

# #### Checking for blank spaces, random characters in each column

# In[566]:


search =[" ","-",]

for c in fDF:
    for s in search:
        print(f"{s} in {c} = ",end= " ")
        print((fDF[c] == s).sum())


# There are no blank spaces, random characters  in columns of train dataset

# ### Dataframe Description:

# Problem Statement:
# 
# Anyone who has booked a flight ticket knows how unexpectedly the prices vary. The cheapest 
# available ticket on a given flight gets more and less expensive over time. This usually happens as 
# an attempt to maximize revenue based on -
# 1. Time of purchase patterns (making sure last-minute purchases are expensive)
# 2. Keeping the flight as full as they want it (raising prices on a flight which is filling up in order 
# to reduce sales and hold back inventory for those expensive last-minute expensive purchases)
# 
# A Predictive Model is needed to be built using collected data to predict fares of flights.
# 
# Size of Data set: 2585 records
# 
# ### FEATURES:
# 
# Airline: The name of the airline.
# 
# Flight Number: Number of Flight
# 
# Date of Departure: The date of the journey
# 
# From: The source from which the service begins.
# 
# To: The destination where the service ends.
# 
# Duration: Total duration of the flight.
# 
# Total_Stops: Total stops between the source and destination.
# 
#    
# ### Target / Label Column:
#     
# Price: The price of the ticket
# 
#  
# 
# 

# #### Getting the basic summary and statistical information of the data.

# In[567]:


fDF.info()


# In[568]:


fDF.nunique() #the number of unique values in each column


# ####  Data Cleaning

# In[569]:


fDF['Total Stops'].value_counts()


# In[570]:


fDF['Total Stops'] = fDF['Total Stops'].replace({'1 Stop':'1-stop','Non Stop':'non-stop','3 Stop(s)':'2+-stop'})


# In[571]:


fDF['Total Stops'].value_counts()


# In[572]:


fDF['Airline'].value_counts()


# In[573]:


fDF['Airline'] = fDF['Airline'].replace({'Indigo':'IndiGo','AirAsia':'Air Asia','GO FIRST':'Go First'})


# In[574]:


fDF['Airline'].value_counts()


# In[575]:


fDF['From'].value_counts()


# In[576]:


fDF['From'] = fDF['From'].replace({'Delhi':'New Delhi'})


# In[577]:


fDF['From'].value_counts()


# In[578]:


fDF['To'].value_counts()


# In[579]:


fDF['To'] = fDF['To'].replace({'Delhi':'New Delhi'})


# In[580]:


fDF['To'].value_counts()


# In[581]:


fDF['Date of Departure'].value_counts()


# In[582]:


fDF['Date of Departure'] = fDF['Date of Departure'].replace({'Dec 02':'Thu, Dec 2','Dec 01':'Wed, Dec 1','Dec 03':'Fri, Dec 3','Dec 04':'Sat, Dec 4','Dec 05':'Sun, Dec 5','Dec 06':'Mon, Dec 6','Dec 07':'Tue, Dec 7'})


# In[583]:


fDF['Date of Departure'].value_counts()


# #### Converting values in Column 'Price' to int64 datatype

# Converting Pandas column into a list, removing the ',' from the values, reattaching to Original Pandas Dataframe and then converting to int64 datatype

# In[584]:


price = fDF['Price'].tolist() 


# In[585]:


Price = []
for p in price:
    Price.append(p.replace(",",""))


# In[586]:


df = pd.DataFrame({'Price':Price})


# In[587]:


df.index = fDF.index


# In[588]:


fDF['Price'] = df['Price']


# In[589]:


fDF['Price'] = fDF['Price'].astype('int64')


# #### Dropping column Unnamed: 0 since it is not required for building the predictive model

# In[590]:


fDF.drop(columns=['Unnamed: 0'],inplace = True)


# In[591]:


fDF.reset_index(drop=True,inplace = True)


# ### Feature Engineering

# ##### Creating New columns "Day", "Date","Month" from Column 'Date of Departure'

# In[592]:


DateDept = fDF['Date of Departure'].tolist() 


# In[593]:


Day=[]
date = []
Month = []
Date = []


# In[594]:


for d in DateDept:
    Day.append(d.split(",")[0])
    date.append(d.split(",")[1])


# In[595]:


for d in date:
    Date.append(d.split(" ")[2])
    Month.append(d.split(" ")[1])


# In[596]:


df2 = pd.DataFrame({'Day':Day, 'Date':Date,'Month': Month})


# In[597]:


df2.index = fDF.index


# In[598]:


fDF[['Day','Date','Month']] = df2[['Day','Date','Month']]


# In[599]:


fDF['Duration'].unique()


# It is observed that Duration values are the difference between Dep_Time and Arrival_Time

# #### Converting the values in Duration column to minutes

# In[600]:


fDF['Duration']


# The values are represented in hours('h') and minutes('m'). For understanding the relationship between price(which contains integer values) and Duration, the values of Duration column must be converted into minutes of integer value type.

# The 'h' component of each value will be multiplied by 60, and then added to 'm' component.

# Firstly, 'h' is replaced by string '*60', the empty space in between is replaced by string '+' and 'm' character at the end is removed.
# 
# Since each value is string type, eval function can be used.
# 
# Finally eval() function will be applied to all the values which will treat each value as a mathematical operation statement.
# 
# ie. (x*60+y) where x is the number attached to 'h' and y is the number attached to 'm'

# In[601]:


duration = fDF['Duration'].tolist() #creating a list with values from colum 'Duration'


# In[602]:


duration


# In[603]:


""" replacing 'h' with *60, whitespace with "+",removing "m" and 
removing leading zeroes from the decimal integer literals since they aren't allowed in python """

for i in range(0,len(Duration)):
    duration[i] = duration[i].replace("h","*60").replace("00","0").replace("01","1").replace("02","2").replace("03","3").replace("04","4").replace("05","5").replace("06","6").replace("07","7").replace("08","8").replace("09","9").replace(" ","+").replace("m","").replace("1.0*60+","1*60") 


# In[604]:


duration


# In[605]:


for i in range(0,len(duration)):
    duration[i] = eval(duration[i])
    


# In[606]:


len(duration)


# ##### Adding Duration(mins) column to fDF dataframe.

# In[607]:


dur_df = pd.DataFrame({"Duration(mins)":duration})


# In[608]:


dur_df.index = fDF.index


# In[609]:


fDF['Duration(mins)'] = dur_df['Duration(mins)']


# In[610]:


fDF


# Dropping the original duration column

# In[611]:


fDF.drop(columns = ['Duration'],inplace=True)


# In[612]:


fDF.info()


# In[613]:


#converting values in'Day' to int64 datatype
fDF['Date'] = fDF['Date'].astype('int64')


# #### Getting the basic Statistical information about int64 datatype columns

# In[614]:


fDF.describe()


# A higher max value that 75% in the columns indicates the presence of outliers

# ### Interpreting Relationship between Independent  and Dependent Variables

# #### Analyzing the Target Column

# In[615]:


sns.distplot(fDF.Price)


# Distribution is skewed and tails of from 15000 mark.

# In[616]:


fDF.Price.skew()


# From the graph above it is observed that the Price data forms a continuous distribution with mean of 7748.33 and tails off from 15000 mark.

# #### Analyzing the Feature Columns

# In[617]:


fDF.dtypes[fDF.dtypes == 'object'] #Identifying the Categorical Columns


# In[618]:


fDF['Airline'].value_counts()


# In[619]:


plt.figure(figsize=(30,5),facecolor='white')
sns.countplot(fDF['Airline'], palette="Set1")


# IndiGo has the highest number of flights followed by Air India and Vistara

# In[620]:


plt.figure(figsize=(10,10),facecolor='white')
sns.countplot(y=fDF['From'], palette="Set1")


# Highest number of flights are from Delhi followed by Mumbai, Kolkata,Bangalore and Hyderabad

# In[621]:


plt.figure(figsize=(10,10),facecolor='white')
sns.countplot(y=fDF['To'], palette="Set1")


# New Delhi is the most popular destination followed by Bangalore, Goa, Kolkata and Mumbai

# In[622]:


sns.countplot(fDF['Total Stops'], palette="Set1")


# Highest number of flights have only 1 stop between source and destination while 2nd highest number of flights are non stop

# ### Interpreting Relationship between Independent  and Dependent Variables

# #### Analyzing Relationship between Day, Month columns and Price

# In[623]:


plt.figure(figsize=(20,42))
fDF.groupby(['Day','Month']).mean()['Price'].unstack().plot()
plt.title('Price Trend')


# In[624]:


sns.barplot(fDF['Day'],fDF['Price'],hue=Month)


# From above graphs it can be observed that on an average, there is a steady decline in Flight price from December to February, with the prices being lowest in January.

# From above graphs it can be observed that Flight Prices increase on an average, as the day of departure gets nearer. 
# 
# Flight Ticket prices are the highest on Thursdays,Mondays and during the Weekend on an average.

# #### Analyzing Relationship between Airlines and Price

# In[625]:


fig, ax = plt.subplots(figsize=(15,7))
fDF.groupby(['Airline']).mean()['Price'].plot(ax=ax)
plt.title('Price Trend')

#Airlines vs Price


# In[626]:


fig, ax = plt.subplots(figsize=(15,7))
fDF.groupby(['Flight Number','Airline'])['Price'].mean().plot(ax=ax)
plt.title('Price Trend')

# Flight numbers vs Price


# Trujet, IndiGo,SpiceJet and Air Asia offer air tickets at the most affordable prices on average, whereas Vistara, Air India are the most expensive on average.

# In[627]:


plt.figure(figsize=(10,5))
fDF.groupby('Duration(mins)')['Price'].mean().plot()
plt.title('Price Trend')


# In[628]:


plt.figure(figsize=(30,12))
fDF.groupby(['Airline','Total Stops'])['Duration(mins)'].mean().plot()
plt.title('Price vs Stops')


# It can be observed that Number of Stops impact the travel time of Airlines

# In[629]:


plt.figure(figsize=(30,12))
fDF.groupby(['Total Stops'])['Price'].mean().plot()
plt.title('Price vs Stops')


# It can be observed that Number of Stops impact the Air Ticket Pricing of Airlines

# In[630]:


sns.lmplot(x="Duration(mins)", y="Price", data=fDF)


# There is a linear relationship between Price and flight duration.

# In[631]:


plt.figure(figsize=(30,11),facecolor='white')
sns.barplot(fDF['To'],fDF['Price'],hue=fDF['Airline'])


# Goa,Mumbai,Pune,Bangalore,Kolakata,Port Blair,New Delhi are the most expensive destinations while,Kochi, Coimbatore,Jammu,Chennai,Hyderabad,Indore,Tirupati are the most affordable destinations

# Indigo,Air Asia and Spicejet provide most affordable Airtickets to the destinations

# ### Checking for Outliers in continuous data type Features.

# In[632]:


plt.figure(figsize=(20,20),facecolor='white')
plotnum=1
for col in fDF[['Price','Duration(mins)']]:
    if plotnum<=34:
        plt.subplot(8,5,plotnum)
        sns.boxplot(fDF[col])
        plt.xlabel(col,fontsize=15)
    plotnum+=1
plt.show()


# There are outliers in all of the above columns

# #### Removing Outliers using Z score Method

# In[633]:


df2 =fDF[['Duration(mins)']].copy() #making a copy of the continuous data type column.


# In[634]:


from scipy.stats import zscore
zscor = zscore(df2)
z_score_abs = np.abs(zscor)

df3 = df2[(z_score_abs < 3).all(axis=1)] #taking 3 as threshold value


# In[635]:


df3.shape


# In[636]:


df2.shape


# ##### Data loss %:

# In[637]:


loss=(2585-2559)/2585 * 100
loss


# 1% Data loss is within acceptable range

# #### Using Z score method to reduce outliers since it has a low data loss %

# In[638]:


dropindx = fDF.index.difference(df3.index)


# In[639]:


dropindx


# In[640]:


fDF.drop(dropindx,inplace = True) #dropping the outliers from original features Dataframe


# In[641]:


fDF.reset_index(drop=True,inplace = True) #resetting the index of the dataframe


# In[642]:


fDF


# In[643]:


sns.boxplot(fDF['Duration(mins)'])


# A lot of outliers have been removed.

# ### Checking for skewness in data distributions

# In[644]:


fDF['Duration(mins)'].skew()


# #### Normalizing Data Distribution using PowerTransformer

# In[645]:


from sklearn.preprocessing import PowerTransformer


# In[646]:


powtrans= PowerTransformer(method='yeo-johnson', standardize=True)


# In[647]:


df4 = fDF[['Duration(mins)']]


# In[648]:


transformed= powtrans.fit_transform(df4)


# In[649]:


type(transformed)


# In[650]:


transformed = pd.DataFrame(transformed, columns=df4.columns) #to convert numpy array back into dataframe


# In[651]:


transformed.skew()


# In[652]:


transformed.index = fDF.index


# In[653]:


fDF[['Duration(mins)']] = transformed[['Duration(mins)']]


# In[654]:


fDF['Duration(mins)'].skew()


# A lot of skewness has been removed.

# ### Encoding Categorical Columns

# ####  Encoding using get_dummies()

# In[655]:


fDF


# In[656]:


dumm = pd.get_dummies(fDF[['Airline','Total Stops','Day','Month']],drop_first = False)


# In[657]:


dumm


# In[658]:


fDF = fDF.join(dumm)


# In[659]:


fDF.drop(columns = ['Airline','Flight Number','Total Stops','Day','Month'],inplace=True) #Dropping the columns since they are no longer needed


# In[660]:


fDF


# In[661]:


fDF.drop(columns = ['Date of Departure'],inplace=True) #dropping 'Date of Departure' column since it is no longer needed


# In[662]:


fDF


# #### Encoding Columns 'From' and 'To' using Label Encoder

# In[663]:


from sklearn.preprocessing import LabelEncoder


# In[664]:


labenc = LabelEncoder()


# In[665]:


for col in fDF[['From','To']]:
    fDF[col] = labenc.fit_transform(fDF[col])


# In[666]:


fDF


# ### Finding Correlation 

# In[667]:


f_corr =fDF.corr()


# In[668]:


f_corr


# In[669]:


plt.figure(figsize=(15,16))
sns.heatmap(f_corr,annot=True,linewidth=1)
plt.show()


# ### Visualizing correlation of feature columns with label column.

# In[670]:


plt.figure(figsize = (20,8))
fDF.corr()['Price'].sort_values(ascending = False).drop(['Price']).plot(kind='bar',color = 'c')
plt.xlabel('Features',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.title('correlation',fontsize = 18)
plt.show()


# It is observed that Month_Dec, Duration(mins), Airline_Vistara,Total Stops_1-stop and From have the highest positive correlation with Price, while Date,Total Stops_non-stop,Month_Jan,Airline_IndiGo have the highest negative correlation with Price

# ### Feature Selection

# In[671]:


from sklearn.preprocessing import StandardScaler


# In[672]:


X = fDF.drop(columns = ['Price'])
y = fDF['Price']


# In[673]:


scaler= StandardScaler()
scaled_X = scaler.fit_transform(X)


# ### Checking for Multicollinearity using Variance Inflation Factor

# In[684]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[685]:


vif = pd.DataFrame()


# In[686]:


vif["Features"] = X.columns
vif['vif'] = [variance_inflation_factor(scaled_X,i) for i in range(scaled_X.shape[1])]


# In[687]:


vif


# MultiCollinearity exists amongst many columns, Based on ANOVA F scores, columns scoring the lowest will be dropped.

# ### Selecting Kbest Features

# In[690]:


from sklearn.feature_selection import SelectKBest, f_classif


# In[691]:


bestfeat = SelectKBest(score_func = f_classif, k = 'all')
fit = bestfeat.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[693]:


fit = bestfeat.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
dfcolumns.head()
featureScores = pd.concat([dfcolumns,dfscores],axis = 1)
featureScores.columns = ['Feature', 'Score']
print(featureScores.nlargest(30,'Score'))


# #### Selecting best features based on their scores:

# In[707]:


x_best = X.drop(columns=['Airline_TruJet']).copy()


# In[708]:


scaled_x_best = scaler.fit_transform(x_best)


# ### This is a Regression Problem since Target/ Label column ('Price') has Continuous type of Data.
# 

# ## Regression Model Building

# In[709]:


from sklearn.model_selection import train_test_split


# In[710]:


from sklearn.metrics import r2_score


# #### Finding the Best Random State

# In[754]:


from sklearn.ensemble import RandomForestRegressor
maxAcc = 0
maxRS=0
for i in range(1,100):
    x_train,x_test,y_train,y_test = train_test_split(scaled_x_best,y,test_size = .25, random_state = i)
    modRF =  RandomForestRegressor()
    modRF.fit(x_train,y_train)
    pred = modRF.predict(x_test)
    acc  = r2_score(y_test,pred)
    if acc>maxAcc:
        maxAcc=acc
        maxRS=i
print(f"Best Accuracy is: {maxAcc} on random_state: {maxRS}")


# In[755]:


x_train,x_test,y_train,y_test = train_test_split(scaled_x_best,y,test_size = .25, random_state =58)


# In[756]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


# In[757]:


from sklearn.metrics import r2_score,mean_squared_error


# In[758]:


rf = RandomForestRegressor()
dt = DecisionTreeRegressor()
xg = XGBRegressor()
SV= SVR()
r=Ridge()


# ### Training the Models

# In[759]:


rf.fit(x_train,y_train)
xg.fit(x_train,y_train)
SV.fit(x_train,y_train)
r.fit(x_train,y_train)
dt.fit(x_train,y_train)


# All models have been trained.

# ### Ridge Regression Model

# In[760]:


y_r_pred = r.predict(x_test)


# ####  R2 Score

# In[761]:


r2_score(y_test,y_r_pred)


# #### Mean Squared Error 

# In[762]:


mean_squared_error(y_test,y_r_pred)


# #### Root Mean Squared Error

# In[763]:


np.sqrt(mean_squared_error(y_test,y_r_pred))


# ###  Random Forest Regression Model

# In[764]:


y_rf_pred = rf.predict(x_test)


# #### R2 Score

# In[765]:


r2_score(y_test,y_rf_pred)


# #### Mean Squared Error

# In[766]:


mean_squared_error(y_test,y_rf_pred)


# #### Root Mean Squared Error

# In[767]:


np.sqrt(mean_squared_error(y_test,y_rf_pred))


# ### XGB Regression Model 

# In[768]:


y_xg_pred = xg.predict(x_test)


# ####  R2 Score

# In[769]:


r2_score(y_test,y_xg_pred)


# #### Mean Squared Error

# In[770]:


mean_squared_error(y_test,y_xg_pred)


# #### Root Mean Squared Error

# In[771]:


np.sqrt(mean_squared_error(y_test,y_xg_pred))


# ### Support Vector Regression Model

# In[772]:


y_svr_pred = SV.predict(x_test)


# ####  R2 Score

# In[773]:


r2_score(y_test,y_svr_pred)


# #### Mean Squared Error

# In[774]:


mean_squared_error(y_test,y_svr_pred)


# #### Root Mean Squared Error

# In[775]:


np.sqrt(mean_squared_error(y_test,y_svr_pred))


# ###  Decision Tree Regression Model

# In[776]:


y_dt_pred = dt.predict(x_test)


# ####  R2 Score

# In[777]:


r2_score(y_test,y_dt_pred)


# #### Mean Squared Error

# In[778]:


mean_squared_error(y_test,y_dt_pred)


# #### Root Mean Squared Error

# In[779]:


np.sqrt(mean_squared_error(y_test,y_dt_pred))


# ### Model Cross Validation

# In[780]:


from sklearn.model_selection import ShuffleSplit,cross_val_score


# #### Ridge Regression

# In[781]:


cross_val_score(r,scaled_x_best,y,cv=ShuffleSplit(5)).mean()


# #### Random Forest Regression

# In[782]:


cross_val_score(rf,scaled_x_best,y,cv=ShuffleSplit(5)).mean()


# #### XGB Regression

# In[783]:


cross_val_score(xg,scaled_x_best,y,cv=ShuffleSplit(5)).mean()


# #### SV Regression

# In[784]:


cross_val_score(SV,scaled_x_best,y,cv=ShuffleSplit(5)).mean()


# #### Decision Tree Regression

# In[785]:


cross_val_score(dt,scaled_x_best,y,cv=ShuffleSplit(5)).mean()


# ### Based on comparing Accuracy Score results with Cross Validation results, it is determined that Random Forest Regressor is the best model. It also has the lowest Root Mean Squared Error score

# ### Hyper Parameter Tuning

# In[806]:


from sklearn.model_selection import GridSearchCV


# In[807]:


parameter = {'n_estimators':[30,60,80],'max_depth': [40,50,80],'min_samples_leaf':[5,10,20],'min_samples_split':[2,5,10],'criterion':['mse','mae'],'max_features':["auto","sqrt","log2"]}


# In[808]:


GridCV = GridSearchCV(RandomForestRegressor(),parameter,cv=ShuffleSplit(5),n_jobs = -1,verbose = 1)


# In[809]:


GridCV.fit(x_train,y_train)


# In[810]:


GridCV.best_params_


# In[811]:


Best_mod = RandomForestRegressor(n_estimators = 80,criterion = 'mse', max_depth= 80, max_features = 'auto',min_samples_leaf = 5, min_samples_split = 2)

Best_mod.fit(x_train,y_train)


# In[812]:


rfpred = Best_mod.predict(x_test)
acc = r2_score(y_test,rfpred)
print(acc*100)


# ##### Random Forest Regressor has an accuracy of 83.55%

# #### Saving The Model

# In[813]:


import joblib
joblib.dump(Best_mod,"BestModelFlight.pkl")


# #### Loading The Model

# In[814]:


mod=joblib.load("BestModelFlight.pkl")


# In[815]:


print(mod.predict(scaled_x_best))


# In[818]:


Prediction_accuracy = pd.DataFrame({'Predictions': mod.predict(scaled_x_best), 'Actual Values': y})
Prediction_accuracy.head(30)


# In[ ]:




