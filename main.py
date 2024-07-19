import pandas as pd
from sklearn.preprocessing import LabelEncoder

## Data Collection and Exploration

# Load the IBM HR Analytics Employee Attrition & Performance Kaggle Dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Print the 1st 5 rows
print(df.head())

## Data Cleaning

# Check if there is any missing value
missing_values = (df.isnull().sum())
print(f'missing_values: \n{missing_values}')
print(f'type of missing_values: {type(missing_values)}')

# As missing_values is a Series which comprises of the no of null values in each column, we find the sum of all values in the Series (sum of no of null values in all columns) and check if the sum is 0
if missing_values.sum() == 0:
    print("No missing values in the dataset")
else:
    print('Some values are missing in the dataset')

## Feature Engineering

# Encode 'Attrition' categorical target variable in strings into numerical values
label_encoder = LabelEncoder()
# Encode 'Yes' to 1 and 'No' to 0 for all values in 'Attrition' column
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# Convert categorical variable into dummy/indicator variables, e.g. OverTime (Yes/ No) column is converted into OverTime_Yes column with True or False boolean values
df = pd.get_dummies(df, drop_first=True)

# To show all columns
pd.set_option('display.max_columns', None)
print(df.head())