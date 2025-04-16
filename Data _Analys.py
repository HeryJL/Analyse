import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import customtkinter as ctk
from tkinter import messagebox

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols


def generer_donnees(n=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 25, size=n),
        'sexe': np.random.choice(['H', 'F'], size=n),
        'heures_etude': np.random.normal(10, 3, size=n),
        'participation': np.random.randint(1, 6, size=n),
        'stress': np.random.normal(5, 2, size=n),
        'activites': np.random.randint(0, 3, size=n),
    })

    data['note'] = (
        2 * data['heures_etude']
        + 1.5 * data['participation']
        - 1.2 * data['stress']
        + np.random.normal(0, 3, size=n)
    )
    return data


def appliquer_acp(data):
    features = ['age', 'heures_etude', 'participation', 'stress', 'activites', 'note']
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = data['note']

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['sexe'] = data['sexe']
    df_pca['note'] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='sexe', palette='Set1')
    plt.title("Projection ACP")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.grid(True)
    plt.show()

    X_pca = df_pca[['PC1', 'PC2']]
    reg = LinearRegression()
    reg.fit(X_pca, y)

    y_pred = reg.predict(X_pca)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print("\n--- Régression sur composantes principales ---")
    print("Coefficients :", reg.coef_)
    print("Intercept :", reg.intercept_)
    print(f"R² : {r2:.3f}")
    print(f"RMSE : {rmse:.2f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, color='dodgerblue', edgecolor='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
    plt.xlabel("Note réelle")
    plt.ylabel("Note prédite")
    plt.title("Régression linéaire sur PC1 & PC2")
    plt.grid(True)
    plt.show()

    return scaler, pca, reg


def appliquer_anova(data):
    data['groupe_etude'] = pd.cut(data['heures_etude'], bins=[0, 9, 20], labels=['faible', 'élevée'])

    print("\n--- ANOVA : Effet du sexe ---")
    model1 = ols('note ~ C(sexe)', data=data).fit()
    print(sm.stats.anova_lm(model1, typ=2))

    print("\n--- ANOVA : Effet du groupe d'étude ---")
    model2 = ols('note ~ C(groupe_etude)', data=data).fit()
    print(sm.stats.anova_lm(model2, typ=2))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='sexe', y='note', data=data)
    plt.title("Note selon le sexe")

    plt.subplot(1, 2, 2)
    sns.boxplot(x='groupe_etude', y='note', data=data)
    plt.title("Note selon le groupe d'étude")
    plt.tight_layout()
    plt.show()


def predire_note(nouvelle_donnee, scaler, pca, reg):
    df_input = pd.DataFrame([nouvelle_donnee])
    X_new = scaler.transform(df_input)
    X_new_pca = pca.transform(X_new)
    prediction = reg.predict(X_new_pca)
    return prediction[0]


def interface_gui():
    data = generer_donnees()
    scaler, pca, reg = appliquer_acp(data)
    appliquer_anova(data)

    def on_predict():
        try:
            age = int(entry_age.get())
            heures_etude = float(entry_etude.get())
            participation = int(entry_participation.get())
            stress = float(entry_stress.get())
            activites = int(entry_activites.get())

            nouvelle_donnee = {
                'age': age,
                'heures_etude': heures_etude,
                'participation': participation,
                'stress': stress,
                'activites': activites,
                'note': 0
            }

            note_predite = predire_note(nouvelle_donnee, scaler, pca, reg)
            messagebox.showinfo("Résultat", f"Note prédite : {note_predite:.2f}")

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("Prédiction de Note avec ACP")
    app.geometry("400x400")

    ctk.CTkLabel(app, text="Âge").pack(pady=5)
    entry_age = ctk.CTkEntry(app)
    entry_age.pack()

    ctk.CTkLabel(app, text="Heures d'étude").pack(pady=5)
    entry_etude = ctk.CTkEntry(app)
    entry_etude.pack()

    ctk.CTkLabel(app, text="Participation (1-5)").pack(pady=5)
    entry_participation = ctk.CTkEntry(app)
    entry_participation.pack()

    ctk.CTkLabel(app, text="Stress (0-10)").pack(pady=5)
    entry_stress = ctk.CTkEntry(app)
    entry_stress.pack()

    ctk.CTkLabel(app, text="Nombre d'activités (0-2)").pack(pady=5)
    entry_activites = ctk.CTkEntry(app)
    entry_activites.pack()

    ctk.CTkButton(app, text="Prédire la note", command=on_predict).pack(pady=20)

    app.mainloop()


if __name__ == "__main__":
    interface_gui()
