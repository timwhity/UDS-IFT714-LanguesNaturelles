##### Table des matières
[Détection d'URLs malicieuses](#détection-durls-malicieuses-ift714)  
[Installation et utilisation](#installation-et-utilisation)  
[Entraîner un modèle](#entraîner-un-modèle)  
[Tester un modèle](#tester-un-modèle)  
[LIME: Interpréter un modèle](#lime-interpréter-un-modèle)  
[Créer ou fusionner un dataset](#créer-ou-fusionner-un-dataset)

# Détection d'URLs Malicieuses (IFT714)
Projet du cours IFT714 en Traitement Automatique des Langues Naturelles. Le projet vise à déterminer automatiquement si une URL est malicieuse ou non et à pouvoir l'expliquer en utilisant LIME (Local Interpretable Model-agnostic Explanations).

Nous avons implémenter les modèles suivants : 
- `roberta`
- `bert`
- `cnn`
- `mlp`
- `decision_tree`

## Installation et utilisation

Pour installer les dépendances, exécutez la commande suivante (nous vous conseillons d'utiliser un environnement virtuel):
```bash
pip install -r requirements.txt
```

Notre projet est constitué de 3 scripts permettant respectivement d'entraîner, d'inférer ou d'interpréter un des modèles présentés ci-haut.

### Entraîner un modèle.

Pour lancer l'entraînement d'un modèle, il suffit d'utiliser le script `train_main.py`. Afin de débuter pour exécuter le code, nous vous suggérons d'entraîner l'arbre de décision sur 1 epoch (question que ce soit rapide):
```bash
python train_main.py dt data/combined_dataset_12/ --model_name decision_tree --num_epochs=1
```
Pour une utilisation plus avancée et pour entraîner d'autres types de modèles, vous pouvez vous référez à l'usage du script ci-dessous. Faire attention à l'argument --limit qui sert principalement pour faire du déboguage sur un système peu performant puisqu'il permet d'interrompre l'entraînement.

```bash
usage: train_main.py [-h]
                     [--num_workers NUM_WORKERS]
                     [--model_name {roberta,decision_tree,cnn,bert,mlp}] 
                     [--batch_size BATCH_SIZE]
                     [--num_epochs NUM_EPOCHS]
                     [--limit LIMIT]
                     experiment_name dataset_directory

Train a model on the URL dataset

positional arguments:
  experiment_name       The name of the experiment
  dataset_directory     The directory containing the dataset already splitted.

options:
  -h, --help            show this help message and exit
  --num_workers NUM_WORKERS
                        The number of workers to use for data loading
  --limit LIMIT         Limit the number of batches for training and testing. Used for CPU testing/training or debugging.
  --model_name [{roberta,decision_tree,cnn,bert,mlp}]
                        The name of the model to use
  --batch_size BATCH_SIZE
                        The batch size for training
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for
```

### Tester un modèle
Pour tester un modèle, il faut tout d'abord avoir entraîné une expérience (soit avoir entraîné un modèle pour lequel un fichier *.pth a été généré dans le dossier models/trained/<experiment_name>). Si ce n'est pas le cas, veuillez vous référez à la section ci-haut pour [entraîner un modèle](#entraîner-un-modèle).

En spécifiant le nom de votre expérience désirée et le bon dataset à utiliser, le script ira automatiquement charger la configuration de cette expérience: quel modèle utiliser ainsi que charger les poids du modèle entraîné. Il exécutera alors votre modèle sur le jeu de données de tests dans le répertoire `dataset_directory/splits`. En sortie, vous obtiendrez une matrice de confusion (en format .png) ainsi que des métriques sur la performance du modèle (en format .json) en test soit l'accuracy, la precision, recall et F1 qui seront sauvegardés dans le dossier correspondant à votre expérience. Voici un exemple d'exécution:

```bash
python test_main.py dt data/combined_dataset_12/
```

Pour un usage plus avancé:

```bash
usage: test_main.py [-h]
                    [--num_workers NUM_WORKERS]
                    [--limit LIMIT]
                    experiment_name dataset_directory

Train a model on the URL dataset

positional arguments:
  experiment_name       The name of the experiment
  dataset_directory     The directory containing the dataset already splitted.

options:
  -h, --help            show this help message and exit
  --num_workers NUM_WORKERS
                        The number of workers to use for data loading
  --limit LIMIT         Limit the number of batches for training and testing. Used for CPU testing/training or debugging.
```

### LIME: Interpréter un modèle

L'usage du script d'interprétation du modèle est similaire à celle du script de test, puisqu'on vient interpréter notre modèle durant son inférence. Par contre, cette fois, au lieu d'inférer les URLs de l'ensemble de tests, on vient exécuter notre modèle à travers LIME. L'ensemble d'URLs à interpréter est prédéfini dans le script lui-même, vous pouvez donc ajouter, modifier ou retirer des URLs comme bon vous semble (nous n'avions pas l'intention de modifier les URLs régulièrement).

Voici un exemple d'exécution:
```bash
python test_lime.py dt data/combined_dataset_12/
```

Pour un usage plus avancé:
```bash
usage: test_lime.py [-h] [--num_workers NUM_WORKERS] [--limit LIMIT] experiment_name dataset_directory

positional arguments:
  experiment_name       The name of the experiment
  dataset_directory     The directory containing the dataset already splitted.

options:
  -h, --help            show this help message and exit
  --num_workers NUM_WORKERS
                        The number of workers to use for data loading
  --limit LIMIT         Limit the number of batches for training and testing. Used for CPU testing/training or debugging.

```

### Créer ou fusionner un dataset

