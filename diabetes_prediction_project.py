import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

"""Data Collection and Analysis

PIMA Diabetes Dataset
"""

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_excel('diabetes.xls', engine='xlrd')

# printing the first 5 rows of the dataset
diabetes_dataset.head()


# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

"""0 --> Non-Diabetic

1 --> Diabetic
"""

diabetes_dataset.groupby('Outcome').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

"""Data Standardization"""

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation
Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

print("Classification Report:\n", classification_report(Y_test, X_test_prediction))

"""Making a Predictive System"""
cm = confusion_matrix(Y_test, X_test_prediction)


input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_df = pd.DataFrame([input_data], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
])

# Standardize the input data
std_data = scaler.transform(input_data_df)
print(std_data)

# Predict using trained classifier
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

#  a figure with 3 subplots
plt.figure(figsize=(15, 5))

# ** Confusion Matrix**
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# ** Outcome Distribution**
plt.subplot(1, 3, 2)
ax = sns.countplot(x="Outcome", data=diabetes_dataset, hue="Outcome", palette="viridis", legend=False)
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = Non-Diabetic, 1 = Diabetic)")
plt.ylabel("Count")

# Add count labels on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

# ** Age Group Analysis**
plt.subplot(1, 3, 3)
# Define age groups
bins = [0, 25, 35, 45, 55, 100]  # Age ranges
labels = ['<25', '25-35', '35-45', '45-55', '55+']  # Labels for groups
#  a new column for Age Group
diabetes_dataset['Age Group'] = pd.cut(diabetes_dataset['Age'], bins=bins, labels=labels)

# Count diabetics in each age group
age_group_analysis = diabetes_dataset.groupby('Age Group', observed=False)['Outcome'].mean() * 100

sns.barplot(x=age_group_analysis.index, y=age_group_analysis.values,
            hue=age_group_analysis.index, palette="coolwarm", legend=False)
plt.title("Diabetes % by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Diabetes Rate (%)")

# Adjust layout
plt.tight_layout()
plt.show()
