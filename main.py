# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Uber Data Analysis Project

# %%
import pandas
import seaborn

# %% [markdown]
# # Load CSV file into memory

# %%
data = pandas.read_csv('uberdata.csv')


# %%
data.tail()


# %%
dt = '4/30/2014 23:22:00'
print(dt)
d,t = dt.split()
print(d,t,sep='\n')
month,date,year = d.split('/')
print(date,month,year,sep='\n')

# %% [markdown]
# # Convert datetime and add some useful columns

# %%
date_time = '4/30/2014 23:22:00'


# %%
date_time= pandas.to_datetime(date_time)


# %%
data['Date/Time'] = data['Date/Time'].map(pandas.to_datetime)


# %%
data.tail()


# %%
def get_dom(date_time):
    return date_time.day

data['dom']=data['Date/Time'].map(get_dom)


# %%
data.tail()


# %%
def get_weekday(date_time):
    return date_time.weekday()
data['Weekday'] = data['Date/Time'].map(get_weekday)

def get_hour(date_time):
    return date_time.hour

data['Hour'] = data['Date/Time'].map(get_hour)


# %%
data.tail()

# %% [markdown]
# # Analysis
# %% [markdown]
# # Datewise Analysis

# %%
import matplotlib.pyplot as plt
plt.hist(data.dom,bins=30,width=.6,range=(.7,30.9))
xticks(range(1,31), data.dom.index)
plt.xlabel('Date of Month')
plt.ylabel ('Frequency')
plt.title ('Frequency by date of month -- uber -- Apr 2014')


# %%
def count_rows(rows):
    return len(rows)

by_date = data.groupby('dom').apply(count_rows)
by_date


# %%
plot(by_date)


# %%
by_date_sorted=(by_date.sort_values())
by_date_sorted


# %%
plt.bar(range(1,31),by_date_sorted)

#xticks(range(1,31), by_date_sorted.index)

# %% [markdown]
# # Analysis according to hour

# %%
plt.hist(data.Hour,bins=24,range=(.5,24),width=.85 , color='g')
plt.title('Analysis according to hour')
plt.xlabel('Hours')
plt.ylabel('Frequency')

# %% [markdown]
# # Analysis according to weekday

# %%
plt.hist(data.Weekday,bins=7,range=(0,6),rwidth=.6,color='r',alpha=.5)

#xticks(range(7), 'Mon Tue Wed Thu Fri Sat Sun'.split())

# %% [markdown]
# # Cross Analysis (hour,weekday)

# %%
cross = data.groupby('Weekday Hour'.split()).apply(count_rows).unstack()


# %%
seaborn.heatmap(cross)


# %%
hist(data['Weekday'],bins=7,range=(0,6),rwidth=.6,color='green')

twiny()
hist(data['Hour'],bins=24,range=(0.5,24),rwidth=.6,color='red',alpha=.5)
("")

# %% [markdown]
# # Analysis by latitude and longitude

# %%
plt.hist(data['Lat'],bins=100,range=(40.5,41),color='y')
("")


# %%
hist(data['Lon'],bins=100,range=(-74.1,-73.9),color='orange')
("")


# %%
hist(data['Lat'],bins=100,range=(40.5,41),color='violet',label='Latitude')
legend(loc='upper left')
twiny()
hist(data['Lon'],bins=100,range=(-74.1,-73.9),color='orange',alpha=.5,label='Longitude')
grid()
legend(loc='upper right')


# %%
plot(data['Lat'])
xlim(0,100)

# %% [markdown]
# # Plotting Linear Regression

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
uber_data = pd.read_csv('Desktop/uber-raw-data-apr14.csv')
uber_data.head()
uber_data.info()
uber_data.describe()
uber_data.columns


# %%
x=[(i) for i in uber_data['Lat']]
y=[(j) for j in uber_data['Lon']]


# %%
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 


# %%
import matplotlib


# %%
plt.plot(x,y, 'yo', x, fit_fn(x))
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Longitude vs Latitude Regression model')


# %%


