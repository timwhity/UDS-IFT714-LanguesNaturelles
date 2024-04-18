# UDS-IFT714-LanguesNaturelles
Projet du cours IFT714 en Traitement Automatique des Langues Naturelles. Le projet vise à déterminer automatiquement si une URL est malicieuse ou non.


## Installation et utilisation

Pour installer les dépendances, exécutez la commande suivante (nous vous conseillons d'utiliser un environnement virtuel) :
```bash
pip install -r requirements.txt
```

Pour lancer le programme, exécutez la commande suivante :
```bash
python train_main.py <experiment_name> <dataset_directory> [--num_workers <n>] [--batch_size <s>] [--num_epochs <e>] [--model_name <m>]
```

Nous avons implémenter les modèles suivants : 
- `roberta`
- `bert`
- `cnn`
- `mlp`
- `decision_tree`