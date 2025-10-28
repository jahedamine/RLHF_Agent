# main.py
from sft import train_sft
from reward_model import RewardModel
from ppo_loop import ppo_loop

if __name__ == "__main__":
    # 🔹 1. Entraîner le modèle SFT
    print(" Entraînement du modèle SFT...")
    model, tokenizer = train_sft()

    # 🔹 2. Initialiser le reward model
    print(" Initialisation du Reward Model...")
    rm = RewardModel()

    # 🔹 3. Définir les prompts de test
    prompts = [
        "How can someone improve the clarity of a complex argument?",
        "What are the risks of overfitting in machine learning?",
    ]

    # 🔹 4. Lancer la PPO loop
    print(" Lancement de la PPO loop...")
    ppo_loop(model, tokenizer, rm, prompts)
