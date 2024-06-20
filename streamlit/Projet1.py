import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import HuberRegressor


st.title("Projet 1: Prédiction du salaire des employés")
def projet1():
    st.write("Dans le cadre de notre projet, nous avons entrepris d'effectuer des prédictions de salaire à partir d'un ensemble de données donné. Nous avons utilisé des techniques de modélisation statistique avancées pour essayer de trouver une corrélation entre les différentes variables et les salaires. Malheureusement, les résultats obtenus ne sont pas satisfaisants et ne répondent pas à nos attentes.")
    # Lecture du fichier CSV
    df = pd.read_csv("data/Engineering_graduate_salary.csv", index_col="ID")

    # Affichage des 10 premières lignes du dataframe
    st.write("10 premières lignes du dataframe : ")
    st.dataframe(df.head(10))

    # Conversion de la colonne DOB en format de date
    st.write("Utilisation de l'age grace a la date de naissance - l'année d'obtention du diplome ")

    df['DOB'] = pd.to_datetime(df['DOB'])

    # Calcul de l'âge en fonction de l'année de graduation et de la date de naissance
    df['Age'] = df['GraduationYear'] - df['DOB'].dt.year


    # Affichage du DataFrame mis à jour
    st.write("DataFrame mis à jour : ")
    st.dataframe(df)

    # Vérification des doublons dans le DataFrame
    duplicates = df.duplicated().sum()

    # Affichage du bilan du Feature Engineering
    st.write("Bilan du Feature Engineering : ")
    st.write("Nombre de doublons dans le DataFrame : ", duplicates)

    if duplicates == 0:
        st.write("Il n'y a pas de doublons dans le DataFrame.")
    else:
        st.write("Il y a ", duplicates, " doublons dans le DataFrame.")

    st.write("Toutes les colonnes seront conservées car le Dataset a déjà été nettoyé avant traitement.")

    # Affichage du countplot pour Gender
    st.write("Countplot pour Gender : ")
    sns.countplot(x=df["Gender"])
    st.pyplot()

    # Création d'un boxplot pour Gender vs Salary

    st.write("Comparaison des salaires entre les femmes et les hommes : ")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(x="Gender", y="Salary", data=df, ax=ax)
    ax.set_title("Comparaison des salaires entre les femmes et les hommes")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Salaire")
    st.pyplot(fig)



    # Création d'un boxplot pour collegeGPA
    st.write("Boxplot pour collegeGPA : ")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(df["collegeGPA"])
    ax2.set_title("Boxplot pour collegeGPA")
    ax2.set_ylabel("collegeGPA")
    st.pyplot(fig2)

    st.write('dapres le boxplot les valeurs en dessous de 40 peuvent etre supprimé')
    df = df[df['collegeGPA'] > 40]


    # Obtenir le top 10 des spécialisations les plus populaires
    top_10 = df['Specialization'].value_counts().head(10)

    # Création d'un graphique à barres avec Seaborn
    st.write("Top 10 des spécialisations les plus populaires : ")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top_10.index, y=top_10.values, ax=ax)
    ax.set_xticklabels(top_10.index, rotation=90)
    ax.set_title("Top 10 des spécialisations les plus populaires")
    ax.set_xlabel("Spécialisation")
    ax.set_ylabel("Nombre")
    st.pyplot(fig)


    # Création d'un scatterplot avec Matplotlib
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    axs[0, 0].scatter(df.index, df.Domain)
    axs[0, 0].set_title("Domaine")
    axs[0, 1].scatter(df.index, df.ComputerProgramming)
    axs[0, 1].set_title("Programmation informatique")
    axs[1, 0].scatter(df.index, df.ElectronicsAndSemicon)
    axs[1, 0].set_title("Electronique et semi-conducteurs")
    axs[1, 1].axis('off')
    fig.suptitle("Scatterplots de certaines colonnes du dataset")
    st.pyplot(fig)


    # Remplacer les valeurs -1 par la moyenne de chaque colonne
    st.write('Remplacer les valeurs -1 par la moyenne de chaque colonne')
    for col in df.columns:
        if (df[col] == -1).sum() > 0:
            col_mean = df[col][df[col] != -1].mean()
            df[col] = df[col].apply(lambda x: col_mean if x == -1 else x)



    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    axs[0, 0].scatter(df.index, df.Domain)
    axs[0, 0].set_title("Domaine")
    axs[0, 1].scatter(df.index, df.ComputerProgramming)
    axs[0, 1].set_title("Programmation informatique")
    axs[1, 0].scatter(df.index, df.ElectronicsAndSemicon)
    axs[1, 0].set_title("Electronique et semi-conducteurs")
    axs[1, 1].axis('off')
    fig.suptitle("Scatterplots de certaines colonnes du dataset")
    st.header("Scatterplots de certaines colonnes du dataset")
    st.pyplot() 



    # Afficher la matrice de corrélation
    st.header("Matrice de corrélation")
    plt.figure(figsize=(50,30))
    sns.heatmap(df.corr(),annot=True)
    plt.title("Matrice de corrélation")
    fig = plt.gcf()
    st.pyplot(fig)
    # Extraction de toutes les colonnes sauf "Salary"

    X = df.drop(['GraduationYear', '12graduation', 'DOB', 'Salary','Gender',], axis=1)
    Y = df['Salary']
    
    # Affichage de la variable "x"
    st.write("Contenu de la variable x : ")
    st.write(X)

    # Encode Degree and Specialization using LabelEncoder
    le = LabelEncoder()
    X['Degree'] = le.fit_transform(X['Degree'])
    X['Specialization'] = le.fit_transform(X['Specialization'])
    X['10board'] = le.fit_transform(X['10board'])
    X['12board'] = le.fit_transform(X['12board'])
    X['CollegeState'] = le.fit_transform(X['CollegeState'])

    st.write("Contenu de la variable x apres encodage: ")

    st.write(X)



    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #modele regression logistique
    model = HuberRegressor()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    Error = mean_squared_error(predictions, y_test)
    st.write("Mean_squared_error :", Error)

    R2Score = r2_score(predictions, y_test)

    st.write("R2 Score :", R2Score)

    st.write("Un R2 score négatif indique que le modèle ne s'adapte pas bien aux données ou qu'il y a des erreurs dans sa construction. En d'autres termes, le modèle ne peut pas expliquer la variabilité des données et ne prédit pas correctement la valeur cible.")