# UDS-IFT714-LanguesNaturelles
Projet du cours IFT714 en Traitement Automatique des Langues Naturelles. Le projet vise à déterminer automatiquement si une URL est malicieuse ou non et à pouvoir l'expliquer en utilisant LIME (Local Interpretable Model-agnostic Explanations).

Nous avons implémenter les modèles suivants : 
- `roberta`
- `bert`
- `cnn`
- `mlp`
- `decision_tree`

## Installation et utilisation

Pour installer les dépendances, exécutez la commande suivante (nous vous conseillons d'utiliser un environnement virtuel) :
```bash
pip install -r requirements.txt
```

Pour lancer le programme, exécutez la commande suivante :
```bash
python train_main.py <experiment_name> <dataset_directory> [--batch_size <s>] [--num_epochs <e>] [--model_name <m>]
```

Vous pouvez également exécuter la commande suivante pour des explications plus détaillées des arguments :
```bash
python train_main.py --help
```

