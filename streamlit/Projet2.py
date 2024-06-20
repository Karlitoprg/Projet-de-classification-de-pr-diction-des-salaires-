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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def projet2():
    # Introduction

    # st.title("Projet N°2 Salaire des Employés")

    st.write("Nous souhaitons entraîner une IA à prédire le salaire des employés en se basant sur plusieurs caractéristiques.")

    list = ["**Age** : L'age des employés", "**Gender** : Le genre des employés", "**Education Level** : Le niveau d'étude des employés", "**Job Title** : Le nom du job des employés", "**Years of Experience** : Les années d'expérience des employés", "**Salary** : Le salaire annuel des employés"]
    st.write("Les colonnes étudiées sont : ")
    for i in list :
        st.markdown("- " + i)


    st.write("Les données peuvent être trouvées sur Kaggle : https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer")

    list2 = ["Analyser les données", "Nettoyer les données", "Modéliser les données", "Entrainer les données", "Conclure"]

    st.write("C'est un problème de régression. Pour le résoudre, nous allons :")

    for n, i in enumerate(list2):
        st.markdown(f"{n+1}. {i}")

    list3 = ["Régression Linéaire","Gradient Boosting","Random Forest", "Régression Ridge", "Régression Lasso"]

    st.write("Comme nous avons des **données numériques** on va utiliser les algorithmes suivants :")
    
    for n, i in enumerate(list3):
        st.markdown(f"{n+1}. {i}")


    # Analyse des données
    st.header("Analyse des données")

    st.write("###  1. Importer les librairies nécessaires")

    st.code("""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso 
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
""")

    

    st.write("### 2. Charger les données dans un DataFrame")


    st.code('''
        df = pd.read_csv("SalaryData.csv")
        df.head()''')
    # Charger les données dans un DataFrame
    df = pd.read_csv("data/SalaryData.csv")

    # Afficher les premières lignes du DataFrame
    st.write("Les premières lignes du DataFrame :")
    st.dataframe(df.head())

    st.write("### 3. Vérifier les doublons")
    # Vérifier les doublons
    st.write(df.duplicated().sum())

    # Vérifier les valeurs nulles

    st.write("### 4. Vérifier les valeurs nulles")
    st.write(df.isnull().sum())

    # Informations du DataFrame
    st.write("### 5. Descriptif du DataFrame :")
    st.write(df.describe())

    st.write("### 6. Divers graphiques à propos du salaire")
    st.write("Données avec un salaire supérieur à 190K $ :")
    st.write(df.loc[(df.Salary > 190000)])


    st.write("Salaire en fonction de l'age :")
    Age = pd.DataFrame(df.groupby('Age')['Salary'].mean())
    st.bar_chart(Age)


    st.write("Salaire en fonction de l'expérience :")
    Experience = pd.DataFrame(df.groupby('Years of Experience')['Salary'].mean())
    st.bar_chart(Experience)

    st.write("Salaire en fonction du niveau d'études")
    Level = pd.DataFrame(df.groupby('Education Level')['Salary'].mean())
    st.bar_chart(Level)

    # Il faudrait mettre le top 10
    st.write("Salaire en fonction du poste")
    jobTitle = pd.DataFrame(df.groupby('Job Title')['Salary'].mean().reset_index())
    jobTitle_sorted = jobTitle.sort_values(by='Salary', ascending=False)[:10]
    jobTitle_sorted.set_index('Job Title', inplace=True)
    st.bar_chart(jobTitle_sorted)



    # Nettoyage des données
    st.header("Nettoyage des données")

    st.write("### 1. Supprimer les doublons")
    # Supprimer les doublons
    st.code("""
        df = df.drop_duplicates()
df.duplicated().sum()
""")

    df.drop_duplicates(inplace=True)
    st.write(df.duplicated().sum())

    st.write("### 2. Supprimer les valeurs nulles")
    df.dropna(inplace=True)
    st.write(df.isnull().sum())

    st.write("### 3. Encoder les colonnes Gender, Education Level et Job Title")
    st.code("""
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Education Level"] = le.fit_transform(df["Education Level"])
    df["Job Title"] = le.fit_transform(df["Job Title"])
        """)

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df["Education Level"] = le.fit_transform(df["Education Level"])
    df["Job Title"] = le.fit_transform(df["Job Title"])
    st.write(df.head())


    # Modélisation des données
    st.write("### 4. Bilan du Feature Engineering")
    st.write("- Les valeurs en **doublons** ont été **supprimées**")
    st.write("- Les valeurs **nulles** ont été **supprimées**")
    st.write("- Les colonnes **Gender**, **Education Level** et **Job Title** ont été **encodées** pour le traitement.")


    st.header("Modélisation des données")

    st.write("### 1. Corrélation entre le salaire et l'âge")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df['Age']
    label = df['Salary']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel('Age')
    plt.ylabel('Salary')
    ax.set_title('Salary vs Age - correlation: ' + str(correlation))
    st.pyplot(fig)

    st.write("Nous constatons que  la corrélation entre les salaires et l\'age est importante. Nous pouvons déduire que l'age a beaucoup d'impact sur les salaires annuels.")

    st.write("### 2. Corrélation entre le salaire et le genre")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df['Gender']
    label = df['Salary']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel('Gender')
    plt.ylabel('Salary')
    ax.set_title('Salary vs Gender - correlation: ' + str(correlation))
    st.pyplot(fig)

    st.write("Nous remarquons une faible corrélation entre le genre et les salaires. Nous pouvons conclure que le genre n'a pas trop d\'influence sur les salaires annuels.  ")

    st.write("### 3. Corrélation entre le salaire et le niveau d'étude")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df['Education Level']
    label = df['Salary']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel('Education Level')
    plt.ylabel('Salary')
    ax.set_title('Salary vs Education Level - correlation: ' + str(correlation))
    st.pyplot(fig)

    st.write("La corrélation entre les salaires et le niveau d\'étude est moyennement haute donc le niveau d\'étude a une influence significative sur les salaires. ")

    st.write("### 4. Corrélation entre le salaire et le job")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df['Job Title']
    label = df['Salary']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel('Job Title')
    plt.ylabel('Salary')
    ax.set_title('Salary vs Job Title - correlation: ' + str(correlation))
    st.pyplot(fig)

    st.write("Nous voyons que la corrélation entre les salaires et les postes est faible. Les postes ont en général peu d'impact sur les salaires mais nous avons constaté deux postes (CEO et CTO) qui ont des salaires élevés.")

    st.write("### 5. Corrélation entre le salaire et les années d'expériences")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df['Years of Experience']
    label = df['Salary']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    ax.set_title('Salary vs Years of Experience - correlation: ' + str(correlation))
    st.pyplot(fig)

    st.write("Nous constatons une trés forte corrélation entre les salaires et les années d'expériences. Nous pouvons en déduire que les années d'expériences ont une influence conséquente sur les salaires annuels. ")

    st.write("### 6. Matrice de corrélation")
    corr_matrix = df.corr()
    fig,ax = plt.subplots()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, ax=ax)
    st.pyplot(fig)


    st.header("Entrainement et optimisation des résultats")

    # séparation des features et des labels
    st.write("### 1. Séparation des features et labels dans 2 variables X et Y")

    st.code("""
        X = df[['Age','Years of Experience', 'Education Level','Job Title']]
        Y = df['Salary']
        """)

    X = df[['Age', 'Gender','Years of Experience', 'Education Level','Job Title',]]
    Y = df['Salary']
    
    # affichage des résultats
    st.write('Features : ')
    st.write(X.head())
    st.write('Labels : ')
    st.write(Y.head())


    # Séparation des données : 75% pour l'entraînement, 25% pour le test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Affichage des données d'entraînement et de test
    st.write("### 2. Séparation des données : 75% pour l'entraînement, 25% pour le test")

    st.code("""
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        """)

    st.write("Taille des données d'entraînement : ", X_train.shape)
    st.write("Taille des données de test : ", X_test.shape)

    st.write("### 3. Régression Linéaire")

    st.code("""
    model = LinearRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(Y_test, predictions)
    """)

    # Entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Prédiction sur les données de test et calcul du coefficient de détermination R2
    predictions = model.predict(X_test)
    r2 = r2_score(Y_test, predictions)

    # Affichage du coefficient de détermination R2
    st.write(f"Le coefficient de détermination R2 est de : {r2:.2f}")

    # Gradient Boosting
    st.write("### 4. Gradient Boosting")

    st.code("""
    model = GradientBoostingRegressor()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    """)

    model = GradientBoostingRegressor()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    st.write(f'R2 score : {r2}')

    # Random Forest
    st.write("### 5. Random Forest")

    st.code("""
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    """)

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    st.write(f'R2 score : {r2}')

    # Régression Ridge
    st.write("### 6. Régression Ridge")

    st.code("""
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    """)
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    st.write(f'R2 score : {r2}')

    # Régression Lasso
    st.write("### 7. Régression Lasso")

    st.code("""
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    """)   

    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(predictions, Y_test)
    st.write(f'R2 score : {r2}')

    st.header('Présentation des résultats')
    st.write('Nous obtenons des scores très satisfaisants (Entre 0.90 et 0.93)')
    st.write("Nous constatons qu'à l'exception de l'âge, toutes les caractéristiques étudiées sont importantes pour prédire le salaire")