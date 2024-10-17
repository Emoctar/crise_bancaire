import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score



# Charger les données
df = pd.read_csv('African_crises_dataset.csv')

#Nettoyer les données
df=df.drop("country" , axis=1)
df=df.drop("country_code", axis=1)
#convertir Gender  en données numeriques
df["banking_crisis"]=df["banking_crisis"].map({"crisis":1, "no_crisis":0})

# Séparer les variables explicatives (X) et la variable cible (y)
X = df.drop('banking_crisis', axis=1)
y = df['banking_crisis']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner le modèle
logreg = LogisticRegression(max_iter=1000, solver='sag')
logreg.fit(X_train_scaled, y_train)

# Évaluer le modèle
y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Sauvegarder le modèle et le scaler
joblib.dump(logreg, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Charger le modèle et le scaler sauvegardés
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Interface utilisateur Streamlit
st.title("Prédiction de crise bancaire en Afrique")

# Créer des champs d'entrée pour chaque caractéristique
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"{feature}", min_value=df[feature].min(), max_value=df[feature].max())

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    # Créer un DataFrame avec les données de l'utilisateur
    data = pd.DataFrame([user_input], columns=X.columns)

    # Normaliser les données
    data_scaled = scaler.transform(data)

    # Faire la prédiction
    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]

    # Afficher le résultat
    if prediction == 1:
        st.success("Risque élevé de crise bancaire")
    else:
        st.success("Risque faible de crise bancaire")

    # Afficher la probabilité
    st.write(f"Probabilité : {proba:.2f}")