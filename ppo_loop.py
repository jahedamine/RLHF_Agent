# ppo_loop.py
import torch

def ppo_loop(model, tokenizer, reward_model, prompts, ppo_steps=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for step in range(ppo_steps):
        print(f"\n PPO Step {step + 1}")
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reward = reward_model.score(prompt, response)
            print(f" Prompt: {prompt}\n Response: {response}\n Reward: {reward:.4f}")

            loss = -torch.tensor(reward, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
