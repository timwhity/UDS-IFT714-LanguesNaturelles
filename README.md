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