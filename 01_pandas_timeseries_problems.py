# BUSINESS SCIENCE UNIVERSITY
# Introduction to Time Series Analysis in Python (PYTIMETK PACKAGE)

# Pytimetk Demo
# Easy, Fast and Fun Time Series Analysis in Python

# NOTES:
# We are using the development version of pytimetk. Installation instructions:
# pip install git+https://github.com/business-science/pytimetk.git

# LIBRARIES ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pytimetk as tk

# 1.0 PROBLEM 1: PANDAS IS TOO COMPLICATED

expedia_df = tk.load_dataset("expedia", parse_dates = ['date_time'])

expedia_df.glimpse()

# 1.1 Pandas Way:
#   6 lines of code, 2 for-loops, 1 if-statement, 1 list comprehension, 1 dictionary comprehension, 1 groupby, 1 resample, 1 agg, 1 reset_index

df_pandas = expedia_df[['site_name', 'date_time', 'cnt', 'is_booking']] \
    .set_index('date_time') \
    .groupby('site_name') \
    .resample('W') \
    .agg({col: ['sum', 'count'] for col in ['cnt', 'is_booking']})

df_pandas.columns = ['_'.join(col).strip() for col in df_pandas.columns.values]

df_pandas.reset_index(inplace = True)

# Pytimetk Way:
#  1 line of code, 1 groupby, 1 summarize_by_time

df_pytimetk = expedia_df[['site_name', 'date_time', 'cnt', 'is_booking']] \
    .groupby('site_name') \
    .summarize_by_time(
        date_column = 'date_time',
        value_column = ['cnt', 'is_booking'],
        freq = 'W',
        agg_func = ['sum', 'count'],
        engine = 'polars' # 13.4x faster than pandas
    )

# SPEED COMPARISON ----

# Pandas
%%timeit -n 10

df_pandas = expedia_df[['site_name', 'date_time', 'cnt', 'is_booking']] \
    .set_index('date_time') \
    .groupby('site_name') \
    .resample('W') \
    .agg({col: ['sum', 'count'] for col in ['cnt', 'is_booking']})

df_pandas.columns = ['_'.join(col).strip() for col in df_pandas.columns.values]

df_pandas.reset_index(inplace = True)

# Polars Engine (Pytimetk)
%%timeit -n 10

df_pytimetk = expedia_df[['site_name', 'date_time', 'cnt', 'is_booking']] \
    .groupby('site_name') \
    .summarize_by_time(
        date_column = 'date_time',
        value_column = ['cnt', 'is_booking'],
        freq = 'W',
        agg_func = ['sum', 'count'],
        engine = 'polars'
    )
    

# PROBLEM 2: PANDAS & MATPLOTLIB IS CODE-HEAVY AND UGLY

# 

df.plot()

# Problem 3 - T






# Plotting
# Calculate the number of rows needed based on unique sites and desired number of columns
num_sites = len(df_pandas['site_name'].unique())
ncols = 5
nrows = -(-num_sites // ncols)  # ceil division

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16), squeeze=False)

# If there are fewer sites than subplots, this will flatten the axes list and iterate only over the number of sites.
axes = axes.ravel()

for ax, (site, group) in zip(axes, df_pandas.groupby('site_name')):
    ax.plot(group['date_time'], group['cnt_sum'], label=f'Site Name {site}')
    ax.set_title(f'Site Name {site}')
    ax.set_xlabel('Date')
    ax.set_ylabel('cnt_sum')
    ax.legend()
    ax.grid(True)

# Turn off any remaining unused subplots
for ax in axes[num_sites:]:
    ax.axis('off')

plt.tight_layout()
plt.show()




df_pytimetk \
    .groupby('site_name') \
    .plot_timeseries(
        date_column = 'date_time',
        value_column = 'cnt_sum',
        facet_ncol = 5,
        width = 1000,
        height = 800,
        title = 'Weekly Bookings by Site Number',
        engine = 'plotly' # plotnine, plotly, or matplotlib
    )
    
    
    