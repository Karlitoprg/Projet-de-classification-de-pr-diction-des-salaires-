import streamlit as st
import Projet2
import Projet1

# st.title('Project On_Boarding')
st.title(":blue[Project IA] :movie_camera:")

# Créer le menu de navigation
menu = ["Projet N°1 Salaire des Etudiants Indien", "Projet N°2 Salaire des Employés"]
choice = st.sidebar.selectbox("Choisissez une page", menu)

# Afficher la première page
if choice == "Projet N°1 Salaire des Etudiants Indien":
    st.header("Projet N°1 Salaire des Etudiants Indien")
    Projet1.projet1()


# Afficher la deuxième page
elif choice == "Projet N°2 Salaire des Employés":
    st.header("Projet N°2 Salaire des Employés")
    Projet2.projet2()
