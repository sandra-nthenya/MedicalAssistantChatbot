import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

# Custom Dataset class
class MedicalDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        encoded_input = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded_output = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].squeeze()
        output_ids = encoded_output['input_ids'].squeeze()
        return {
            'input_ids': input_ids,
            'labels': output_ids
        }

# Load and preprocess data
try:
    data = pd.read_csv('data/medical_dialogues.csv')
    questions = data['short_question'].tolist()
    answers = data['short_answer'].tolist()
except KeyError as e:
    print(f"Column not found: {e}")
    raise

# Load pre-trained model and tokenizer
model_name = "microsoft/BioGPT"  # Use the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create dataset
dataset = MedicalDataset(questions, answers, tokenizer)

# # Define a custom data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     model=model,
#     padding=True,
# )

# Set training arguments
training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./models/medical_chatbot')
tokenizer.save_pretrained('./models/medical_chatbot')
