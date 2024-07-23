import mlflow
import json
import random
from collections import defaultdict

mlflow.set_experiment("/Users/praty@dashworks.ai/inranker-ft-dashworks-queries")
from inranker import InRankerTrainer
from datasets import Dataset, DatasetDict

trainer = InRankerTrainer(
    model="unicamp-dl/InRanker-small",
    warmup_steps=1000,
    batch_size=8,
    gradient_accumulation_steps=1,
    bf16=True, # If you have a GPU with BF16 support
    output_dir="trained_model_logging_ws1000_bf16_e30_val_ml2560_split_queries",
    save_steps=500,
    num_train_epochs=30,
    logging_steps=100,
    max_length=2560,
)

# Load the full dataset
full_dataset = trainer.load_custom_dataset(
    distill_file="inranker_training_data_jul18.jsonl", max_length=2560
)

# Group data by queries
query_to_data = defaultdict(list)
for idx in range(len(full_dataset)):
    item = full_dataset[idx]
    query_to_data[item['query']].append(idx)

# Get unique queries
unique_queries = list(query_to_data.keys())

# Randomly select 10% of queries for validation
validation_size = int(0.1 * len(unique_queries))
validation_queries = set(random.sample(unique_queries, validation_size))

# Create training and validation datasets
train_indices = []
validation_indices = []

for query, indices in query_to_data.items():
    if query in validation_queries:
        validation_indices.extend(indices)
    else:
        train_indices.extend(indices)

# Use select to create new datasets
train_dataset = full_dataset.select(train_indices)
validation_dataset = full_dataset.select(validation_indices)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(validation_dataset)}")

# Train the model
trainer.train(train_dataset=train_dataset, validation_dataset=validation_dataset)
