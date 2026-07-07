\# Sources des données



\## UrbanSound8K

\- URL : https://urbansounddataset.weebly.com/urbansound8k.html

\- Auteurs : J. Salamon, C. Jacoby, J. P. Bello (2014)

\- Téléchargé via : Kaggle API (`kaggle datasets download -d chrisfilo/urbansound8k`)

\- Licence : usage académique/recherche uniquement

\- Classes utilisées : `siren` (classID 8) comme exemples positifs

\- ⚠️ Ne pas reshuffler les folds — utiliser validation croisée 10-fold à l'entraînement

\- Date de téléchargement : 07/07/2026



\## ESC-50

\- URL : https://github.com/karoldvl/ESC-50

\- Auteur : Karol J. Piczak

\- Licence : Creative Commons Attribution-NonCommercial

\- Téléchargé via : `git clone https://github.com/karoldvl/ESC-50.git data/raw/esc50`

\- Classes utilisées :

&#x20; - Positifs supplémentaires : `siren`

&#x20; - Négatifs (bruit ambiant) : `car\_horn`, `engine`, `rain`, `wind`, `street\_music`

\- Date de téléchargement : 07/07/2026

