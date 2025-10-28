# sft.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import SFTTrainer
import torch

def train_sft(model_name="gpt2", dataset_name="LDJnr/LessWrong-Amplify-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset(dataset_name)
    def format_example(example):
        input_text = example["conversation"][0]["input"]
        output_text = example["conversation"][0]["output"]
        return {"text": f"### Instruction:\n{input_text}\n\n### Response:\n{output_text}"}
    formatted_dataset = dataset["train"].map(format_example)

    training_args = TrainingArguments(
        output_dir="./lesswrong-agent-gpt2",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    class PrintLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                print(f"🧠 Step {state.global_step} | Loss: {logs['loss']:.4f}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        formatting_func=lambda x: x["text"],
        callbacks=[PrintLossCallback()],
    )

    trainer.train()
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_sft()
    print("✅ SFT terminé. Modèle prêt.")
