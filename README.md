# 🧠 GenRL++ Agent — Modular RLHF from Scratch

## 🔍 Vision

Ce projet n’est pas une simple implémentation de RLHF.  
C’est une œuvre modulaire, introspective, et transmissible.  
Chaque brique est visible, chaque choix est documenté, chaque gradient est un acte d’alignement.

---

## 📚 Dataset utilisé

Le modèle SFT est entraîné sur [`LDJnr/LessWrong-Amplify-Instruct`](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct), un corpus issu de la communauté LessWrong.

Ce choix reflète une volonté d’aligner l’agent sur des réponses claires, nuancées, et argumentées.  
Le dataset contient des paires `Instruction` / `Response` structurées, idéales pour un fine-tuning supervisé.

> “Avant d’apprendre à juger, l’agent apprend à parler avec clarté.”

---

## 📦 Structure du projet

├── sft.py # Fine-tuning supervisé (SFT) 
├── reward_model.py # Reward model avec .score() 
├── ppo_loop.py # Boucle PPO manuelle
├── utils.py # Fonctions auxiliaires 
├── main.py # Exécution du pipeline complet 
├── requirements.txt # Dépendances



---

## 🔁 Pipeline

1. **SFT** : Entraînement sur LessWrong
2. **RM** : Reward model scalaire
3. **PPO** : Optimisation par renforcement
4. **Audit** : Affichage des prompts, réponses, rewards

---

## 🧪 Fonctionnalités actuelles

- Génération de réponses via SFT
- Scoring scalaire via RewardModel
- PPO loop manuelle avec backpropagation
- Logs clairs et auditables

---

## 🔜 Extensions prévues
ce projet est une version initiale du projet GenRL qui aura comme amelioration les extensions suivantes ensuite :
- RM miroir (jugement par posture)
- Juges multiples (clarté, style, profondeur)
- Mémoire longue
- Justification introspective
- README narratif version agent
- Benchmark philosophique

---

## 🧠 Auteur

Projet conçu et structuré par **jahed Amine** 
Chaque ligne de code est un acte de clarté 
chaque README un miroir de posture.

> “Je ne veux pas juste un agent qui fonctionne."  
> "Je veux un agent qui me ressemble.”
