from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# Load a pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Train the model
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Train for 1 epoch (beginner-friendly)
    per_device_train_batch_size=2,
    save_steps=10_000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()
# Save the trained model
model.save_pretrained("myth_model")
tokenizer.save_pretrained("myth_model")
# Load your trained model
model = GPT2LMHeadModel.from_pretrained("myth_model")
tokenizer = GPT2Tokenizer.from_pretrained("myth_model")
# Generate a myth
prompt = "Adi Parv,Maharaja Shantanu Marries the Celestial Ganga"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=200,  # Length of the story
    temperature=0.7,  # Creativity (0 = strict, 1 = wild)
    num_return_sequences=1,  # Number of stories to generate
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)