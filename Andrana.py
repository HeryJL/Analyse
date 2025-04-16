# Projet d'Analyse de Données avec ANOVA et Régression Linéaire

# 1. Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Configuration
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
np.random.seed(42)

# 2. Création du jeu de données
n = 200  # nombre d'étudiants
methods = ['Traditionnel', 'En ligne', 'Hybride']
method_effect = {'Traditionnel': 0, 'En ligne': 5, 'Hybride': 8}

data = {
    'Methode': np.random.choice(methods, size=n),
    'Temps_Etude': np.random.normal(10, 2, n),
    'Note': np.zeros(n)
}

for i in range(n):
    meth = data['Methode'][i]
    temps = data['Temps_Etude'][i]
    data['Note'][i] = (50 + method_effect[meth] + 2 * temps + np.random.normal(0, 5))

df = pd.DataFrame(data)
print(df)
# 3. Analyse exploratoire
print("=== Statistiques descriptives ===")
print(df.describe())

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='Methode', y='Note', data=df)
plt.title('Distribution des notes par méthode')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Temps_Etude', y='Note', hue='Methode', data=df)
plt.title('Relation temps d\'étude - note')
plt.tight_layout()
plt.show()

# 4. Analyse ANOVA
print("\n=== Analyse ANOVA ===")
model_anova = ols('Note ~ C(Methode)', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print(anova_table)
beta0,beta1 = 50.38, 2.04
for i in range(n):
    data['Note'][i] = beta0 + beta1*data['Temps_Etude'][i]

df = pd.DataFrame(data)

# 5. Régression linéaire simple
print("\n=== Régression linéaire simple ===")
X = sm.add_constant(df[['Temps_Etude']])
xx = df['Temps_Etude']
y = df['Note']

model_reg = sm.OLS(y, X).fit()
print(model_reg.summary())

plt.figure(figsize=(10, 6))
sns.regplot(x='Temps_Etude', y='Note', data=df, ci=95)
plt.title('Régression: Temps d\'étude vs Note')
plt.show()

# 6. Régression linéaire multiple
print("\n=== Régression linéaire multiple ===")
df_dummies = pd.get_dummies(df, columns=['Methode'], drop_first=True)
X_multi = sm.add_constant(df_dummies[['Temps_Etude', 'Methode_En ligne', 'Methode_Hybride']])
model_multi = sm.OLS(df_dummies['Note'], X_multi).fit()
print(model_multi.summary())

# 7. Validation du modèle
X_train, X_test, y_train, y_test = train_test_split(X_multi.drop('const', axis=1), df_dummies['Note'], test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\n=== Performance du modèle ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Analyse des résidus')
plt.show()

# 8. Interprétation
print("\n=== Interprétation ===")
print("1. ANOVA:")
print(f"Les différences entre méthodes sont {'significatives' if anova_table['PR(>F)'][0] < 0.05 else 'non significatives'} (p={anova_table['PR(>F)'][0]:.4f})")

print("\n2. Régression simple:")
print(f"Chaque heure d'étude supplémentaire augmente la note de {model_reg.params['Temps_Etude']:.2f} points")

print("\n3. Régression multiple:")
print(f"- Méthode en ligne: +{model_multi.params['Methode_En ligne']:.2f} vs traditionnel")
print(f"- Méthode hybride: +{model_multi.params['Methode_Hybride']:.2f} vs traditionnel")
print(f"- Temps d'étude: +{model_multi.params['Temps_Etude']:.2f} par heure")