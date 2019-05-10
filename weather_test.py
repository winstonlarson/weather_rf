
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

#%%
temps = pd.read_csv('temps.csv')

#%%
years = temps.year
months = temps.month
days = temps.day

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in 
         zip(years, months, days)]
dates =[datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

#%%
fig, ((plot1, plot2), (plot3, plot4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
fig.autofmt_xdate(rotation=45)

plot1.plot(dates, temps['actual'])
plot1.set_xlabel('')
plot1.set_ylabel('Temperature')
plot1.set_title('Max temp')

plot2.plot(dates, temps['temp_1'])
plot2.set_xlabel('')
plot2.set_ylabel('')
plot2.set_title('One day prior')

plot3.plot(dates, temps['temp_2'])
plot3.set_xlabel('Date')
plot3.set_ylabel('Temperature')
plot3.set_title('Two days prior')

plot4.plot(dates, temps['friend'])
plot4.set_xlabel('Date')
plot4.set_ylabel('')
plot4.set_title('Friend\'s guess')
plt.style.use('fivethirtyeight')

plt.tight_layout(pad=2)

#%%
temps_adj = temps[temps['temp_2']<100]
temps_adj = temps_adj[temps_adj['temp_1']<100]

dumb_temps = temps_adj.drop(['year'],axis=1)
smart_temps = temps_adj.drop(['year', 'week'],axis=1)

#%%
dumb_temps = pd.get_dummies(dumb_temps)

#%%
dumb_labels = np.array(dumb_temps['actual'])
dumb_features = np.array(dumb_temps.drop('actual', axis=1))
smart_labels = np.array(smart_labels['actual'])
smart_features = np.array(smart_features.drop('actual', axis=1))

#%%
dumb_train_features, dumb_test_features, dumb_train_labels, dumb_test_labels = train_test_split(
    dumb_features, dumb_labels, test_size=0.25, random_state=42)

smart_train_features, smart_test_features, smart_train_labels, smart_test_labels = train_test_split(
    smart_features, smart_labels, test_size=0.25, random_state=42)

