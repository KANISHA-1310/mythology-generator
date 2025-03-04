import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load dataset (you need to actually load a dataset first)
# Here's an example using Hugging Face Datasets
dataset = datasets.load_dataset("csv", data_files={"train": r"C:\Users\HP\my_project\ai_myth_generator\mahabharath_1-2.csv"})  # Update path

# Print column names to debug
print("Column names:", dataset["train"].column_names)

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    # Ensure the input is a string
    texts = [str(text) for text in examples["Summary"]]
    return tokenizer(texts, 
                    truncation=True, 
                    max_length=128,
                    padding="max_length")

# Debug: Print the first few examples to check the data
print("First few examples:", dataset["train"][:5])

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # More reasonable for learning
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train and save
trainer.train()
model.save_pretrained("myth_model")
tokenizer.save_pretrained("myth_model")

# Generation function
def generate_myth(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=300,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
print(generate_myth("According to the historical records of this earth"))