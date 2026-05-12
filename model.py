import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\data (1).csv")

X = df.drop(['diagnosis', 'id'], axis=1)

X = X.dropna(axis=1)
X = X.astype('float64')


X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

pca = PCA()

X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=[f'PC{i+1}' for i in range(X_scaled.shape[1])])
print(X_pca)

print(X.corr())
print(X_pca.corr())

y= df['diagnosis']

from sklearn.model_selection import train_test_split    

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

print(pca.explained_variance_ratio_[:20].sum())

X_final = X_pca.iloc[:, :20]
print(X_final)

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y, test_size=0.3, random_state=42)

model_final = LogisticRegression()
model_final.fit(X_train_final, y_train_final)

print(model_final.score(X_test_final, y_test_final))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid()
plt.show()

import joblib
joblib.dump(model_final, 'logistic_regression_model.pkl')



