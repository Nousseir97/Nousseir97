import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

''' Read gender_submission CSV '''

example = pd.read_csv('gender_submission.csv')
print()
print(example.head(10))

'''Load the data into dataframe'''

df = pd.read_csv('train.csv')

# ''' Summary statistics '''
#
# print(df.describe())

'''columns list'''

print(df.columns)



''' Summary of total null values for all features in dataframe '''

print(df.isnull().sum())


''' Treating missing values in the 'Age' column using imputation via the mean '''

df['Age'] = df['Age'].replace(np.NaN, df['Age'].mean())

#print(df.isnull().sum())



''' Label encoding categorical columns '''

''' Columns and their datatypes '''

print(df.dtypes)

''' create an instance of the class '''

labelencoder = LabelEncoder()

''' List only categorical features that need to be encoding '''

titanic_categorical_features = ['Sex']

''' an efficient way to label encode any number of categorical variables '''

def label_encoding(column):

    df[f'{column}_N'] = labelencoder.fit_transform(df[f'{column}'])

    return df

''' The code below is to check the resulting dataframe '''

for i in map(label_encoding, titanic_categorical_features):
    print(i)



''' Feature Engineering, e.g. linear combination of features '''

df['Family Size'] = df['SibSp'] + df['Parch']
#print(df)



''' Choose target variable '''

y = df.Survived

''' Choose features '''

titanic_features = ['Pclass', 'Sex_N', 'Age', 'Family Size']

X = df[titanic_features]

''' Define the model '''

titanic_model = DecisionTreeRegressor(random_state = 1)

''' Fit the model '''

titanic_model.fit(X, y)



''' Predict on first 5 rows of training dataset '''
#
# print(X.head())
# initial_test = titanic_model.predict(X.head())
# print(initial_test)



''' Survival prediction on test data '''


''' Load test data into dataframe '''

df_test = pd.read_csv('test.csv')



''' Treating missing values in the 'Age' column using imputation via the mean '''

df_test['Age'] = df_test['Age'].replace(np.NaN, df_test['Age'].mean())

#print(df_test.isnull().sum())



''' Label encoding categorical columns '''

test_categorical_features = ['Sex']

def label_encoding_2(column):

    df_test[f'{column}_N'] = labelencoder.fit_transform(df_test[f'{column}'])

    return df_test

for i in map(label_encoding_2, test_categorical_features):
   print(i)



''' Feature engineering '''

df_test['Family Size'] = df_test['SibSp'] + df_test['Parch']



''' Chosen features'''

test_features = ['Pclass', 'Sex_N', 'Age', 'Family Size']

val_X = df_test[test_features]

survival_predictions = titanic_model.predict(val_X)

print(survival_predictions)


''' A new dataframe assigning survival predictions for each passenger '''

result = pd.DataFrame(df_test['PassengerId'])
result.insert(1, "Survived", survival_predictions, True)
result["Survived"] = pd.to_numeric(result["Survived"], downcast="integer")

result.to_csv("titanic_survival_predictions.csv", index=False)


