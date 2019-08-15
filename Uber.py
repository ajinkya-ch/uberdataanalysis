
# coding: utf-8

# In[1]:


print('hello jupyter notebook')


# # Uber Data Analysis Project

# In[27]:


import pandas
import seaborn


# # Load CSV file into memory

# In[26]:


data = pandas.read_csv('Desktop/uber-raw-data-apr14.csv')


# In[4]:


data.tail()


# In[5]:


dt = '4/30/2014 23:22:00'
print(dt)
d,t = dt.split()
print(d,t,sep='\n')
month,date,year = d.split('/')
print(date,month,year,sep='\n')


# # Convert datetime and add some useful columns

# In[6]:


date_time = '4/30/2014 23:22:00'


# In[7]:


date_time= pandas.to_datetime(date_time)


# In[8]:


data['Date/Time'] = data['Date/Time'].map(pandas.to_datetime)


# In[9]:


data.tail()


# In[10]:


def get_dom(date_time):
    return date_time.day

data['dom']=data['Date/Time'].map(get_dom)


# In[11]:


data.tail()


# In[12]:


def get_weekday(date_time):
    return date_time.weekday()
data['Weekday'] = data['Date/Time'].map(get_weekday)

def get_hour(date_time):
    return date_time.hour

data['Hour'] = data['Date/Time'].map(get_hour)


# In[13]:


data.tail()


# # Analysis

# # Datewise Analysis

# In[14]:


plt.hist(data.dom,bins=30,width=.6,range=(.7,30.9))
("")
xticks(range(1,31), data.dom.index)
plt.xlabel('Date of Month')
plt.ylabel ('Frequency')
plt.title ('Frequency by date of month -- uber -- Apr 2014')


# In[15]:


def count_rows(rows):
    return len(rows)

by_date = data.groupby('dom').apply(count_rows)
by_date


# In[16]:


plot(by_date)


# In[17]:


by_date_sorted=(by_date.sort_values())
by_date_sorted


# In[18]:


plt.bar(range(1,31),by_date_sorted)

xticks(range(1,31), by_date_sorted.index)
("")


# # Analysis according to hour

# In[19]:


plt.hist(data.Hour,bins=24,range=(.5,24),width=.85 , color='g')
plt.title('Analysis according to hour')
plt.xlabel('Hours')
plt.ylabel('Frequency')


# # Analysis according to weekday

# In[20]:


plt.hist(data.Weekday,bins=7,range=(0,6),rwidth=.6,color='r',alpha=.5)

xticks(range(7), 'Mon Tue Wed Thu Fri Sat Sun'.split())
("")


# # Cross Analysis (hour,weekday)

# In[21]:


cross = data.groupby('Weekday Hour'.split()).apply(count_rows).unstack()


# In[22]:


seaborn.heatmap(cross)


# In[23]:


hist(data['Weekday'],bins=7,range=(0,6),rwidth=.6,color='green')

twiny()
hist(data['Hour'],bins=24,range=(0.5,24),rwidth=.6,color='red',alpha=.5)
("")


# # Analysis by latitude and longitude

# In[32]:


plt.hist(data['Lat'],bins=100,range=(40.5,41),color='y')
("")


# In[25]:


hist(data['Lon'],bins=100,range=(-74.1,-73.9),color='orange')
("")


# In[26]:


hist(data['Lat'],bins=100,range=(40.5,41),color='violet',label='Latitude')
legend(loc='upper left')
twiny()
hist(data['Lon'],bins=100,range=(-74.1,-73.9),color='orange',alpha=.5,label='Longitude')
grid()
legend(loc='upper right')
("")


# In[27]:


plot(data['Lat'])
xlim(0,100)


# # Plotting Linear Regression

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


uber_data = pd.read_csv('Desktop/uber-raw-data-apr14.csv')
uber_data.head()
uber_data.info()
uber_data.describe()
uber_data.columns


# In[4]:


x=[(i) for i in uber_data['Lat']]
y=[(j) for j in uber_data['Lon']]


# In[6]:


fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 


# In[10]:


import matplotlib


# In[20]:


plt.plot(x,y, 'yo', x, fit_fn(x))
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Longitude vs Latitude Regression model')

