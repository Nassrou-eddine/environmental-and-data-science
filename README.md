# environmental-and-data-science
favoriser l'échange de connaissances et d'outils, d'encourager la collaboration entre les disciplines et de promouvoir l'utilisation de la science des données pour des initiatives environnementales durables.

# # 1/Analyse des Déchets Plastiques
Ce script Python utilise la bibliothèque Pandas pour analyser un jeu de données intitulé "Plastic Waste Around the World". Il commence par charger les données et explorer leur structure à l'aide de diverses méthodes telles que head(), describe(), et info() pour obtenir un aperçu des colonnes, des valeurs manquantes, et des doublons.

Ensuite, des visualisations sont réalisées avec Matplotlib et Seaborn pour examiner la distribution des risques liés aux déchets côtiers et les principales sources de déchets plastiques. Des graphiques en secteurs et des histogrammes permettent de visualiser les proportions de chaque catégorie de risque et des sources principales.

Le script utilise également GeoPandas pour créer des cartes thématiques montrant la distribution des déchets plastiques par habitant, le taux de recyclage, et le risque côtier, en fusionnant les données du jeu avec des informations géographiques.

En outre, il implémente des modèles de machine learning (Random Forest et CatBoost) pour prédire les risques côtiers et les principales sources de déchets. Des techniques de prétraitement telles que l'encodage des étiquettes et la normalisation des caractéristiques sont appliquées avant la séparation des données en ensembles d'entraînement et de test. Enfin, le script évalue la performance des modèles à l'aide de la précision et d'autres métriques.

Cette approche intégrée permet non seulement d'analyser les données des déchets plastiques, mais également de fournir des insights exploitables pour les politiques de gestion des déchets et les initiatives de durabilité.
