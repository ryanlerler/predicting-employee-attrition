import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


## Data Collection and Exploration

# Load the IBM HR Analytics Employee Attrition & Performance Kaggle Dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Print the 1st 5 rows
print(df.head())


## Data Cleaning

# Check if there is any missing value
missing_values = df.isnull().sum()
# print(f'missing_values: \n{missing_values}')
# print(f'type of missing_values: {type(missing_values)}')

# As missing_values is a Series which comprises of the no of null values in each column, we find the sum of all values in the Series (sum of no of null values in all columns) and check if the sum is 0
if missing_values.sum() == 0:
    print("No missing values in the dataset\n")
else:
    print('Some values are missing in the dataset')


## Feature Engineering

# Encode 'Attrition' categorical target variable in strings into numerical values
label_encoder = LabelEncoder()
# Encode 'Yes' to 1 and 'No' to 0 for all values in 'Attrition' column
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

# Convert categorical variable into dummy/indicator variables, e.g. OverTime (Yes/ No) column is converted into OverTime_Yes column with True (1) or False (0) values
df = pd.get_dummies(df, drop_first=True)

# To show all columns in the 1st 5 rows
# pd.set_option('display.max_columns', None)
print(df.head())


## Split of dataset into training and testing sets

# Remove 'Attrition' column from the dataset (axis=0 to remove row, axis=1 to remove column)
# X is the input variables or features that are used to predict the target variable
X = df.drop('Attrition', axis=1)

# y is the target variable to predict, i.e. employee attrition
y = df['Attrition']

# Split the dataset (1470 * 20% = 294 testing sets; 1176 training sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


## Selection, training, and evaluation of model

# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

# The warning indicates that the logistic regression model did not converge within the specified number of iterations. This can happen if the model struggles to find the optimal solution given the data
# We can increase the max_iter parameter to allow the model more iterations to converge.
# We should scale all features so that they all have similar ranges and can help the optimization process converge more easily.
# Different solvers have different convergence properties. We can try using a different solver, such as 'liblinear', 'sag', or 'saga'.

# Scaling transforms data so that it fits within a specific range, often between 0 and 1
# Features in a dataset may have different units and ranges (e.g., age might be between 0 and 100, while income might be in the range of thousands). Algorithms that compute distances or gradients, like logistic regression, can be biased by features with larger ranges. Scaling ensures that all features contribute equally to the result.
# Scaling can help optimization algorithms converge faster. If features have very different scales, the algorithm might take longer to find the optimal solution.

# Standardize the features such that they each have a standard deviation of 1 and a mean of 0
scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# UserWarning: X has feature names, but LogisticRegression was fitted without feature names
# Create DataFrame from the ndarray and then pass feature names to the scaled data to address the warning above
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Select and train the model
# Model Fitting is a measurement of how well a machine learning model adapts to data that is similar to the data on which it was trained. The fitting process is generally built-in to models and is automatic. A well-fit model will accurately approximate the output when given new data, producing more precise results.
clf = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
print(f"Score without scaling: {clf.score(X_test, y_test)}")
print(f"Score with scaling: {clf.score(X_test_scaled, y_test)}")

# Evaluate the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Mean Squared Error: {mse}')


## Model Interpretation and Visualization

# Explain the importance of each feature using coefficients
feature_importance = pd.DataFrame(clf.coef_[0], index=X_train.columns, columns=['Importance'])
print(feature_importance.sort_values(by='Importance', ascending=False))

# Results below show that working overtime is the largest factor that leads to employee attrition at IBM:
#             Importance
# OverTime_Yes                         0.974048
# JobRole_Laboratory Technician        0.781314
# BusinessTravel_Travel_Frequently     0.715211
# YearsAtCompany                       0.676929
# MaritalStatus_Single                 0.631571
# NumCompaniesWorked                   0.507755
# JobRole_Sales Representative         0.505918
# YearsSinceLastPromotion              0.499597
# BusinessTravel_Travel_Rarely         0.438683
# JobRole_Research Scientist           0.397902
# JobRole_Sales Executive              0.385753
# JobRole_Human Resources              0.361483
# DistanceFromHome                     0.341057
# Department_Sales                     0.284338
# Gender_Male                          0.220425
# MaritalStatus_Married                0.213290
# MonthlyIncome                        0.203316
# JobRole_Manufacturing Director       0.165116
# EducationField_Technical Degree      0.125824
# MonthlyRate                          0.096520
# Education                            0.069070
# Department_Research & Development    0.048908
# HourlyRate                           0.035788
# JobRole_Manager                      0.020408
# EmployeeCount                        0.000000
# StandardHours                        0.000000
# PerformanceRating                   -0.024124
# EmployeeNumber                      -0.054096
# PercentSalaryHike                   -0.056098
# EducationField_Marketing            -0.096955
# DailyRate                           -0.114581
# EducationField_Other                -0.124325
# JobLevel                            -0.125052
# StockOptionLevel                    -0.169623
# RelationshipSatisfaction            -0.176990
# TrainingTimesLastYear               -0.206171
# Age                                 -0.247040
# WorkLifeBalance                     -0.255298
# EducationField_Medical              -0.261286
# JobRole_Research Director           -0.279008
# EducationField_Life Sciences        -0.295541
# JobInvolvement                      -0.340619
# EnvironmentSatisfaction             -0.412458
# JobSatisfaction                     -0.434021
# TotalWorkingYears                   -0.460259
# YearsWithCurrManager                -0.466027
# YearsInCurrentRole                  -0.647474

# Plot a confusion matrix to visualize model performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Make predictions for all employees
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
predictions = clf.predict(X_scaled)
probabilities = clf.predict_proba(X_scaled)[:, 1]

# Add predictions and probabilities to the dataframe
df['Attrition_Prediction'] = predictions
df['Attrition_Probability'] = probabilities

# Identify employees likely to resign (predicted attrition = 1)
employees_likely_to_resign = df[df['Attrition_Prediction'] == 1]
print(employees_likely_to_resign[['EmployeeNumber', 'Attrition_Probability']])

# Save the updated DataFrame to a new CSV file
df.to_csv('Updated_Employee_Attrition.csv', index=False)