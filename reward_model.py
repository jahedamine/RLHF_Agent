# reward_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RewardModel(torch.nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.linear = torch.nn.Linear(self.model.config.n_embd, 1)

    def score(self, prompt, response):
        full_input = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            score = self.linear(last_hidden[:, -1, :])
        return score.item()

if __name__ == "__main__":
    rm = RewardModel()
    score = rm.score("What is RLHF?", "RLHF is reinforcement learning with human feedback.")
    print(f"🏆 Score: {score:.4f}")
