# Conception et implémentation d’un algorithme d’analyse de sentiment basé sur les aspects 


Ce projet est une implémentation d'algorithmes d'analyse des sentiments basé sur les aspects en Python. Ils visent à déterminer la polarité des sentiments (positif, négatif ou neutre) d'aspects spécifiques mentionnés dans les données textuelles. Le projet est structuré en divers scripts, chacun correspondant à différentes étapes du pipeline d'analyse, notamment le prétraitement, l'identification de polarité, la visualisation des données, le calcul des sentiments basé sur des règles et l'approche d'apprentissage automatique.

## Pour Commencer

Pour exécuter ce projet, vous aurez besoin de Python installé sur votre système. Il est recommandé d'utiliser Visual Studio Code (VSCode) comme environnement de développement intégré (IDE) pour gérer et exécuter les scripts de manière pratique.

### Prérequis

Avant d'exécuter les scripts, vous devez installer les bibliothèques requises. Vous pouvez les installer à l'aide de la commande suivante :

```bash
pip install nltk pandas scikit-learn matplotlib
```

De plus, vous devrez peut-être télécharger des ressources spécifiques à partir de la bibliothèque NLTK. Cela peut être fait dans l'interpréteur Python ou en modifiant les scripts pour inclure *nltk.download('resource_name')*.

### Exécuter les scripts
Pré-traitement (*pre-processing.py*) : tokenise les phrases, attribue des balises part-of-speech et effectue la reconnaissance d'entités nommées.

**Éxécuter**:
```bash
python pre-processing.py
```

Ou simplement run sur Visual Studio Code.


Identificateur de polarité (*polarity-identifier.py*) : identifie la polarité des termes d'aspect à l'aide de SentiWordNet.

**Éxécuter**:
```bash
python polarity-identifier.py
```

Visualisation des données (*data-visualization.py*) : visualise la répartition des sentiments dans les fichiers traités.

**Éxécuter**:
```bash
python data-visualization.py
```

Approche à règles (*aspectTermsPolarity-RulebasedApproach.py*) : calcule le sentiment des termes d'aspect en fonction des mots environnants et de leur poids grammatical.

**Éxécuter**:
```bash
python aspectTermsPolarity-RulebasedApproach.py
```

Approche d'apprentissage automatique (*aspectTermsPolarity-MachineLearningApproach.py*) : utilise un RandomForestClassifier pour prédire le sentiment des termes d'aspect.

**Éxécuter**:
```bash
python aspectTermsPolarity-MachineLearningApproach.py
```

Chaque script reçoit en entrées des fichiers XML et les traite en conséquence. Assurez-vous que les fichiers de données XML sont placés dans le répertoire correct comme indiqué dans les scripts ou ajustez les chemins de fichiers dans les scripts si nécessaire.

