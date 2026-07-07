# 🔊 A.E.D.A.S. — Acoustic Emergency Detection & Alert System

<p align="center">
  <img src="https://img.shields.io/badge/Status-En%20cours-orange?style=for-the-badge&logo=git" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Environment-SIL%20%28Software--in--the--Loop%29-green?style=for-the-badge" alt="Environment"/>
</p>

---

### 🚨 Système embarqué de détection acoustique de véhicules prioritaires

**A.E.D.A.S.** est un système innovant de détection de véhicules prioritaires (ambulances, pompiers, police) basé sur la reconnaissance audio en temps réel de leurs sirènes. Le système estime également la direction d'approche du véhicule et simule des actions correctives de sécurité au sein du véhicule hôte (baisse du volume audio, alerte visuelle sur tableau de bord virtuel).

Le projet est développé dans un environnement **entièrement logiciel et simulé (Software-in-the-Loop - SIL)**. Aucun matériel physique n'est requis pour valider la faisabilité technique et évaluer les performances du système.

---

## 🎯 Objectifs Clés

*   **🧠 Détection IA en Temps Réel :** Identification précise des sirènes à partir de flux audio simulés à l'aide de Deep Learning.
*   **📡 Estimation de la Direction (TDOA) :** Calcul de l'angle d'approche via la méthode de différence de temps d'arrivée (*Time Difference of Arrival*) sur des microphones virtuels.
*   **🎛️ Actions Correctives & Sécurité :** Simulation des réactions de l'habitacle, telles que la diminution automatique du volume audio de l'autoradio et l'affichage d'une alerte dynamique sur une interface homme-machine (HMI) virtuelle.
*   **⚡ Optimisation Embarquée :** Optimisation et quantification du modèle (INT8 via TensorFlow Lite Micro) pour un déploiement simulé sur microcontrôleur (Cortex-M via Renode/QEMU).

---

## 🏗️ Architecture du Système

Le système est découpé en **4 couches logiques** simulant une chaîne de traitement embarquée complète :

| Couche | Technologie / Mécanisme | Rôle |
| :--- | :--- | :--- |
| **1. Acquisition Audio (Simulée)** | Rejeu multi-canal, `librosa`, sons réels | Mixage et simulation d'un environnement sonore routier bruyant avec sirène |
| **2. Traitement IA (Émulé)** | Extraction MFCC, CNN 1D quantisé INT8 | Analyse spectrale et classification en temps réel des signaux audio |
| **3. Estimation Directionnelle** | Calcul TDOA, corrélation croisée | Détermination de l'angle d'arrivée (Azimut) sur l'ensemble de micros |
| **4. Actions & Communication** | Bus CAN simulé (`python-can`), Streamlit | Transmission des alertes de sécurité et affichage sur le Dashboard |

---

## 📁 Structure du Répertoire

```text
aedas/
├── data/
│   ├── raw/          # Données audio brutes (non versionnées)
│   └── processed/    # Features extraites, datasets nettoyés et normalisés
├── notebooks/         # Explorations, R&D et prototypage Jupyter
├── src/               # Code source modulaire et réutilisable
├── docs/              # Documentation technique, diagrammes et spécifications
├── reports/           # État de l'art, rapports de performance et rapport final
├── requirements.txt   # Dépendances Python du projet
└── README.md          # Guide principal du projet
```

---

## 🛠️ Stack Technique

Le projet repose sur un écosystème logiciel moderne et robuste :

*   **Traitement du Signal & IA :** `Python 3.11`, `NumPy`, `SciPy`, `librosa`, `TensorFlow` / `Keras`, `TensorFlow Lite Micro`, `scikit-learn`.
*   **Simulation & Embarqué :** `Renode` / `QEMU` (émulation Cortex-M), `python-can` (simulation de réseau de bord CAN).
*   **Visualisation / HMI :** `Streamlit` / `Dash` pour le tableau de bord interactif de démonstration.
*   **Analyse de données & Dev :** `Pandas`, `Matplotlib`, `Jupyter Lab` (pour le prototypage rapide).

---

## 🚀 Installation & Démarrage rapide

### Prérequis
Assurez-vous d'avoir **Python 3.11** installé sur votre système.

### Cloner & Configurer l'environnement

1.  **Cloner le dépôt**
    ```bash
    git clone <url-du-repo>
    cd aedas
    ```

2.  **Créer un environnement virtuel** (ex: sous Windows)
    ```bash
    py -3.11 -m venv venv
    ```

3.  **Activer l'environnement virtuel**
    *   **Windows (PowerShell) :**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Windows (CMD) :**
        ```cmd
        .\venv\Scripts\activate.bat
        ```
    *   **Linux / macOS :**
        ```bash
        source venv/bin/activate
        ```

4.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚧 Statut du Projet

Le projet est actuellement **en cours de développement actif** (Phase de prototypage et de traitement du signal). Les contributions au pipeline de traitement IA et à l'estimation directionnelle TDOA sont en cours de structuration.
