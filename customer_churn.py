#Import thư viện
import pandas as pd    #Using for data manipulation
import pymssql         #Using for connect to mssql server
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

"""# Data Overview

"""

#Define stats server
Server = '45.117.83.230'
Port = 1433
Account=  'Student_DA_Q1'
Password = '@MindXDream2023'
db = 'DA_FINALTEST'
#Create connection to server
connection = pymssql.connect(host = Server, port = Port, user = Account, password = Password, database = db)
query = 'SELECT * FROM [dbo].[Customer_Churn_Banker]'

df = pd.read_sql(query, connection)  #Query data from server to python
df.head()

df.shape

df.info()

"""# Data Cleaning"""

# Check NA value
df.isnull().sum()

# Check duplicated data
df.duplicated().sum()

df.describe()

"""# EDA"""

num_col = []
for col in df.columns:
    if df[col].dtype != 'object' and col not in ['customer_id', 'credit_card', 'active_member', 'churn']:
        num_col.append(col)
fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (14, 20))
for i, column in enumerate(num_col):
        row = i//2
        col = i%2
        ax = axes[row, col]
        sns.histplot(data= df[column], ax = ax, kde= True)
plt.show()

for i in num_col:
    sns.histplot (data = df, x= i, kde = True, hue = 'churn' )
    plt.show()

sns.catplot(data=df, x="churn", y="products_number", kind="violin")
sns.catplot(data=df, x="churn", y="balance", kind="violin")

cat_col = [i for i in df.columns if df[i].dtype == 'object']
cat_col

plt.figure(figsize = (10, 20))
sns.catplot(x = 'country',data = df, kind = "count", hue = 'churn', palette = "Blues",  height=4, aspect=2)
plt.title('Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

plt.figure(figsize = (10, 20))
sns.catplot(x = 'credit_card',data = df, kind = "count", hue = 'churn', palette = "Blues",  height=4, aspect=2)
plt.title('Credit Card')
plt.xlabel('Credit Card')
plt.ylabel('Count')
plt.show()

plt.figure(figsize = (10, 20))
sns.catplot(x = 'active_member',data = df, kind = "count", hue = 'churn', palette = "Blues",  height=4, aspect=2)
plt.title('Active Members')
plt.xlabel('Active Members')
plt.ylabel('Count')
plt.show()

plt.figure(figsize = (10, 20))
sns.catplot(x = 'gender',data = df, kind = "count", hue = 'churn', palette = "Blues",  height=4, aspect=2)
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

"""# Encoding"""

df['gender'] = df['gender'].astype(str)

df['gender'] = df['gender'].map({'Male':1, 'Female':0})

df['country'].unique()
range(df['country'].nunique())
df['country'] = df['country'].replace(df['country'].unique(), range(df['country'].nunique()))

df

correlation = df.corr()
plt.figure(figsize=(10,7))
corr_map = sns.heatmap(correlation, annot=True)

"""# Models

Logistic Regression
"""

X=df.drop(columns=['churn']).values
y=df['churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in kf.split(X, y):
    print(f'Fold:{fold}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    fold += 1

# import model
model_lg = LogisticRegression()
# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = model.predict(X_test)
score = cross_validate(model_lg, X, y, cv=kf, scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'], return_estimator=True)
score
# Đánh giá mô hình
# 1. Độ chính xác (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
# 2. Báo cáo phân loại (Classification Report)
report = classification_report(y_test, y_pred)
print(report)
# 3. AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
# 4.Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_pred_prob = model_lg.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
# Vẽ ROC curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.3f)' % roc_auc, color="r")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Logistic Regression ROC Curve', fontsize=16)
plt.legend(loc="lower right")
plt.show()

"""Support Vector Classification"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
# Chọn các đặc điểm và biến mục tiêu
X = df.drop(columns=['churn']).values
y = df['churn'].values
# Chuẩn hóa đặc điểm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa đặc điểm
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Tạo và huấn luyện mô hình SVM
model = SVC(kernel='linear')  # Experiment with different kernels
model.fit(X_train, y_train)
# Dự đoán trên tập dữ liệu kiểm tra
y_pred = model.predict(X_test)
# Đánh giá mô hình
# 1. Độ chính xác (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
# 2. Báo cáo phân loại (Classification Report)
report = classification_report(y_test, y_pred)
print(report)
# 3. AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
# 4.Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", confusion)

"""Xgboost"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
# Chọn các đặc điểm và biến mục tiêu
X = df.drop(columns=['churn']).values
y = df['churn'].values
# Chuẩn hóa đặc điểm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa đặc điểm
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Tạo và huấn luyện mô hình XGBoost
model = XGBClassifier()
model.fit(X_train, y_train)
# Dự đoán trên tập dữ liệu kiểm tra
y_pred = model.predict(X_test)
# Đánh giá mô hình
# 1. Độ chính xác (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
# 2. Báo cáo phân loại (Classification Report)
report = classification_report(y_test, y_pred)
print(report)
# 3. AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
# 4.Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
# Chọn các đặc điểm và biến mục tiêu
X = df.drop(columns=['churn']).values
y = df['churn'].values
# Chuẩn hóa đặc điểm
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa đặc điểm
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Tạo và huấn luyện mô hình LightGBM
model = LGBMClassifier()
model.fit(X_train, y_train)
# Dự đoán trên tập dữ liệu kiểm tra
y_pred = model.predict(X_test)
# Đánh giá mô hình

# 1. Độ chính xác (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
# 2. Báo cáo phân loại (Classification Report)
report = classification_report(y_test, y_pred)
print(report)

# 3. AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)

# 4.Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

"""**Nhận xét:** Mô hình có độ chính xác cao nhât khi sử dụng LightGBM(87%), ROC curve và chỉ số AUC cho thấy mô hình logistic regression có hiệu suất tốt trong việc phân biệt các trường hợp dương tính với các trường hợp âm tính. Tuy nhiên, vẫn cần cải thiện độ chính xác của mô hình"""