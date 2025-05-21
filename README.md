# data-analytics-project-
This dataset from Stats NZ shows weighted average prices of various food items in New Zealand for September 2023. It reflects national consumption patterns and regional variations, supporting inflation analysis and economic planning. Ideal for analysts, it's publicly available for use in data visualization, trend analysis, and policy insights.


print(data_file.head())  # print the first few rows of the dataframe
print(data_file.info())  # print a concise summary of the dataframe
print(data_file.describe())  # print a summary of the central tendency and variability of the dataframe
print(data_file.columns)  # print the column names of the dataframe
print(data_file.isnull().sum())  # print the number of missing values in each column

Series_reference  ...  Series_title_1
0     CPIM.SAP0100  ...    Oranges, 1kg
1     CPIM.SAP0100  ...    Oranges, 1kg
2     CPIM.SAP0100  ...    Oranges, 1kg
3     CPIM.SAP0100  ...    Oranges, 1kg
4     CPIM.SAP0100  ...    Oranges, 1kg

[5 rows x 8 columns]
None
             Period    Data_value
count  29683.000000  29601.000000
mean    2015.154881      5.600928
std        4.837254      4.843288
min     2006.060000      0.900000
25%     2011.070000      2.700000
50%     2015.090000      3.770000
75%     2019.100000      6.340000
max     2023.090000     43.510000
Index(['Series_reference', 'Period', 'Data_value', 'STATUS', 'UNITS',
       'Subject', 'Group', 'Series_title_1'],
      dtype='object')
Series_reference     0
Period               0
Data_value          82
STATUS               0
UNITS                0
Subject              0
Group                0
Series_title_1       0
dtype: int64

Thought: Now that we have a basic understanding of the data, let's visualize the distribution of the Data_value column to gain more insights. We can use a histogram to visualize the distribution. Additionally, we can plot the Data_value column against the Period column to see if there are any trends or patterns.

![histogram_Data_value](https://github.com/user-attachments/assets/e5c8bb17-7f7c-40e5-a16d-20e8a150d7f0)
![lineplot_Data_value_over_Time](https://github.com/user-attachments/assets/41b5ecbb-4a7d-4cff-b796-f01088b96481)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot a histogram of the Data_value column
plt.clf()
sns.histplot(data_file['Data_value'].dropna(), kde=True)
plt.title('Distribution of Data_value')
plt.xlabel('Data_value')
plt.ylabel('Frequency')
plt.savefig('./figures/histogram_Data_value.png')

# Plot the Data_value column against the Period column
plt.clf()
sns.lineplot(x='Period', y='Data_value', data=data_file.dropna())
plt.title('Data_value over Time')
plt.xlabel('Period')
plt.ylabel('Data_value')
plt.savefig('./figures/lineplot_Data_value_over_Time.png')




To provide insights into the uploaded file and make future predictions, I'll follow a different approach.

First, let's analyze the data to understand the relationships between columns. We can use correlation analysis to identify which columns are highly correlated with the 'Data_value' column.

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation between columns
correlation_matrix = data_file[['Period', 'Data_value']].corr()

# Print the correlation matrix
print(correlation_matrix)

This will give us an idea of how the 'Period' column is related to the 'Data_value' column.

Next, we can use a linear regression model to make predictions about future values of 'Data_value'. We'll train the model on the existing data and then use it to predict values for future periods.

from sklearn.linear_model import LinearRegression
import numpy as np

# Create a linear regression model
model = LinearRegression()

# Define the feature (Period) and target (Data_value) columns
X = data_file[['Period']]
y = data_file['Data_value']

# Train the model on the data
model.fit(X, y)

# Generate future period values
future_periods = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)

# Make predictions for future periods
future_predictions = model.predict(future_periods)

# Print the future predictions
print(future_predictions)

Finally, we can plot the historical data along with the future predictions to visualize the trends and predictions.

plt.clf()
plt.plot(data_file['Period'], data_file['Data_value'], label='Historical Data')
plt.plot(future_periods, future_predictions, label='Future Predictions', linestyle='--', marker='o')
plt.title('Data_value over Time')
plt.xlabel('Period')
plt.ylabel('Data_value')
plt.legend()
plt.savefig('./figures/historical_and_future_data.png')

This plot will give us a clear understanding of the historical trends and the predicted future values of 'Data_value'.

By following this approach, we've analyzed the data, made predictions about future values, and visualized the results to provide insights into the uploaded file.
