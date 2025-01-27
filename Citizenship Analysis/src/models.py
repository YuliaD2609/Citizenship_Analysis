import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Addestramento dei modelli
def train_models(X, y):
    # Divisione dei dati in set di addestramento e di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dati divisi fra i set")

    # Standardizzazione delle caratteristiche
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    print("Dati standardizzati")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(max_iter=50, max_depth=3),
        'Decision Tree': DecisionTreeClassifier(max_depth=3)
    }

    trained_models = {}
    for model_name, model in models.items():
        print(model_name)
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
    print("Modelli salvati")

    save_path = os.path.join('../Models')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for model_name, model in trained_models.items():
        joblib.dump(model, (os.path.join(save_path, f'{model_name}.pkl')))
    print("Modelli salvati nella cartella")

    return trained_models, X_test_scaled, y_test

