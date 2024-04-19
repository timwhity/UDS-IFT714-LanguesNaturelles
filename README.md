##### Table des matières
[Détection d'URLs malicieuses](#détection-durls-malicieuses-ift714)  
[Installation et utilisation](#installation-et-utilisation)  
[Entraîner un modèle](#entraîner-un-modèle)  
[Tester un modèle](#tester-un-modèle)  
[LIME: Interpréter un modèle](#lime-interpréter-un-modèle)  
[Créer ou fusionner un dataset](#créer-ou-fusionner-un-dataset)
[Structure du projet](#structure-du-projet)

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
usage: test_lime.py [-h]
                    [--num_workers NUM_WORKERS]
                    [--limit LIMIT]
                    experiment_name dataset_directory

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
Si vous souhaitez créer votre propre dataset ou vous souhaitez refaire un split d'un des datasets existants, alors vous aurez besoin du script `split_data.py` qui vous permet de split ou de combiner des ensembles de données CSV. L'argument positionnel "dataset" peut prendre plusieurs chemins vers différents datasets. Si plusieurs datasets sont listés, alors le script fera automatiquement la combinaison entre les datasets, retirera les doublons et génèrera des statistiques par rapport à celui-ci.

Une caractéristique implémentée, mais non-incluse dans les résultats est la possibilité de pouvoir balancer les ensembles de données. Comme ceux-ci favorisent généralement un plus grand nombre d'URLs bénignes, on offre la possibilité de sous-échantillonner la classe la plus fréquente de sorte à ce que les deux classes soient représentées équitablement en spécifiant l'argument `--balance`.

Voici un exemple d'exécution pour séparer un ensemble de données:
```bash
python split_data.py data/test/urldata.csv --save_dir=data/test/
```

Pour un usage plus avancé:
```bash
usage: split_data.py [-h]
                     [--balance]
                     [--test_ratio TEST_RATIO]
                     [--seed SEED]
                     --save_dir SAVE_DIR
                     dataset [dataset ...]

positional arguments:
  dataset               Path to the single dataset CSV.

options:
  -h, --help            show this help message and exit
  --balance             Balance the dataset.
  --save_dir SAVE_DIR   Directory to save the split datasets.
  --test_ratio TEST_RATIO
                        Validation ratio from the training set.
  --seed SEED
```

## Structure du projet

```
╗
╠══ README.md : Explication du projet
╠══ requirements.txt : Fichier contenant les dépendances du projet
╠══ train_main.py : Script principal du projet permettant de lancer l'entraînement
╠══ test_main.py : Script permettant de lancer les tests
╠══ test_lime.py : Script permettant de lancer les prédictions de LIME
║
╠══╦══ data/ : Dossier contenant les données
║  ╠══ combined_dataset_12/ : Dossier contenant le dataset combiné
║  ╠══ dataset_1/ : Dossier contenant le dataset 1
║  ╠══ dataset_2/ : Dossier contenant le dataset 2
║  ╠══ data_utils.py : Script contenant des fonctions pour charger les données
║  ╠══ dataset.py : classe UrlDataset pour charger les données
║  ╠══ feature_extractor.py : classe FeatureExtractor pour le chargement des données de DecisionTree
║  ╠══ split_data_features.py : Script pour séparer les données en features et labels
║  ╚══ stats_dataset.py : Script pour afficher les statistiques des datasets
║
╠═════ docs/ : Dossier contenant la documentation et les rapports du projet
║
╠══╦══ interpretability/ : Dossier contenant les scripts pour l'interprétabilité
║  ╚══ url_explainer.py : Script contenant la classe UrlExplainer pour expliquer les prédictions
║
╠══╦══ models/ : Dossier contenant les modèles
║  ╠══ trained/ : Dossier contenant les modèles entraînés
║  ╠══ bert.py : Classe pour le modèle BERT
║  ╠══ cnn.py : Classe pour le modèle CNN
║  ╠══ decision_tree.py : Classe pour le modèle Decision Tree
║  ╠══ mlp.py : Classe pour le modèle MLP
║  ╠══ model.py : Classe abstraite pour les modèles
║  ╚══ roberta.py : Classe pour le modèle RoBERTa
║
╠══╦══ trainers/ : Dossiers des classes réalisant l'entraînement des modèles
║  ╠══ base_trainer.py : Classe abstraite pour les entraîneurs
║  ╠══ bert_trainer.py : Classe pour l'entraîneur de BERT
║  ╠══ cnn_trainer.py : Classe pour l'entraîneur de CNN
║  ╠══ decision_tree_trainer.py : Classe pour l'entraîneur de Decision Tree
║  ╠══ mlp_trainer.py : Classe pour l'entraîneur de MLP
║  ╠══ roberta_trainer.py : Classe pour l'entraîneur de RoBERTa
║  ╚══ trainer_metrics.py : Classe pour les métriques des entraîneurs
║
╚══╦══ utils/ : Dossier contenant des utilitaires
   ╠══ basic_tokenizer.py : Classe pour le tokenizer de base
   ╠══ model_utils.py : Fonctions utilitaires pour les modèles
   ╠══ torch_utils.py : Fonctions utilitaires pour PyTorch
   ╚══ utils.py : Fonctions utilitaires
```
