# A.E.D.A.S. — Acoustic Emergency Detection & Alert System

Système embarqué de détection de véhicules prioritaires (ambulances, pompiers, police) par
reconnaissance audio de sirènes, avec estimation de la direction d'approche et simulation
d'actions correctives (baisse du volume audio, alerte sur tableau de bord virtuel).

Le projet est développé en environnement **entièrement logiciel et simulé**
(Software-in-the-Loop) : aucun matériel physique n'est requis pour valider la faisabilité
technique et les performances du système.

## Objectifs

- Détecter en temps réel des sirènes à partir de flux audio simulés
- Estimer la direction d'approche via TDOA (Time Difference Of Arrival) sur microphones virtuels
- Simuler les actions correctives associées : baisse automatique du volume audio, affichage
  d'une alerte sur un mock HMI
- Optimiser un modèle de deep learning pour un déploiement embarqué simulé (Cortex-M / QEMU)

## Architecture

| Couche | Description |
|---|---|
| 1. Acquisition audio (simulée) | Rejeu multi-canal de sons réels (sirènes + bruit routier) |
| 2. Traitement IA (émulé) | Extraction MFCC + classification par CNN 1D quantisé INT8 |
| 3. Estimation directionnelle | Calcul TDOA en post-traitement sur signaux simulés |
| 4. Actions & communication (simulées) | Bus CAN simulé + interface graphique (dashboard) |

## Structure du repo

aedas/
├── data/
│   ├── raw/          # Données audio brutes (non versionnées)
│   └── processed/    # Features extraites, datasets nettoyés
├── notebooks/         # Explorations et prototypage Jupyter
├── src/               # Code source réutilisable
├── docs/              # Documentation technique
├── reports/           # Rapport d'état de l'art, rapport final
├── requirements.txt
└── README.md

## Installation

```bash
git clone <url-du-repo>
cd aedas
py -3.11 -m venv venv
venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

## Technologies

Python, NumPy, librosa, SciPy, TensorFlow/Keras, TensorFlow Lite Micro, scikit-learn,
Renode/QEMU (simulation embarquée), python-can (simulation bus CAN), Streamlit/Dash
(dashboard de démonstration).

## Statut

🚧 Projet en cours de développement.