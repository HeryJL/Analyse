# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# 1. Préparation des données
# Données d'exemple (heures d'étude vs notes)
data = {
    'Heures_etude': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Notes': [45, 55, 65, 70, 75, 80, 85, 88, 90, 92]
}

# Création d'un DataFrame
df = pd.DataFrame(data)
print("=== Données initiales ===")
print(df.head())

# 2. Analyse exploratoire
print("\n=== Statistiques descriptives ===")
print(df.describe())

# Visualisation des données
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['Heures_etude'], df['Notes'], color='blue')
plt.title('Relation heures d\'étude / notes')
plt.xlabel('Heures d\'étude')
plt.ylabel('Notes')
plt.grid(True)

# 3. Régression linéaire
X = df['Heures_etude'].values.reshape(-1, 1)  # Variable indépendante
y = df['Notes'].values                         # Variable dépendante

# Création et entraînement du modèle
model = LinearRegression()
model.fit(X, y)

# Prédictions
y_pred = model.predict(X)

# Coefficients du modèle
print("\n=== Résultats de la régression ===")
print(f"Coefficient (pente): {model.coef_[0]:.2f}")
print(f"Interception: {model.intercept_:.2f}")
print(f"Équation de la droite: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")

# Métriques d'évaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"\nMSE (Mean Squared Error): {mse:.2f}")
print(f"R² (Coefficient de détermination): {r2:.2f}")

# Visualisation de la régression
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, y_pred, color='red', linewidth=2, label='Régression linéaire')
plt.title('Régression linéaire')
plt.xlabel('Heures d\'étude')
plt.ylabel('Notes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Analyse des résidus
residuals = y - y_pred

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Analyse des résidus')
plt.xlabel('Heures d\'étude')
plt.ylabel('Résidus')
plt.grid(True)

# 5. Prédiction pour de nouvelles valeurs
heures_nouvelles = np.array([2.5, 4.5, 7.5, 11]).reshape(-1, 1)
predictions = model.predict(heures_nouvelles)

print("\n=== Prédictions ===")
for heures, note in zip(heures_nouvelles, predictions):
    print(f"{heures[0]} heures d'étude → Note prédite: {note:.1f}")

# 6. Analyse statistique avancée
n = len(y)
x_mean = np.mean(X)
Sxx = np.sum((X - x_mean)**2)
se = np.sqrt(np.sum(residuals**2) / (n - 2))  # Erreur standard des résidus

# Intervalle de confiance à 95% pour la pente
t_value = stats.t.ppf(1 - 0.025, df=n-2)
ci = t_value * se / np.sqrt(Sxx)
print(f"\n=== Analyse statistique ===")
print(f"Intervalle de confiance à 95% pour la pente: {model.coef_[0]:.2f} ± {ci:.2f}")

# Test d'hypothèse pour la pente
t_stat = model.coef_[0] / (se / np.sqrt(Sxx))
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
print(f"Test d'hypothèse (H0: pente = 0): t = {t_stat:.2f}, p-value = {p_value:.4f}")