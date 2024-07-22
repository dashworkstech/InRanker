import mlflow
import json

mlflow.set_experiment("/Users/praty@dashworks.ai/inranker-ft-dashworks-queries")
from inranker import InRankerTrainer
from datasets import DatasetDict

trainer = InRankerTrainer(
    model="unicamp-dl/InRanker-small",
    warmup_steps=1000,
    batch_size=8,
    gradient_accumulation_steps=1,
    bf16=True, # If you have a GPU with BF16 support
    output_dir="trained_model_logging_ws200_bf16_e30_val_ml2560",
    save_steps=500,
    num_train_epochs=30,
    logging_steps=100,
    max_length=2560,
)

train_dataset = trainer.load_custom_dataset(
    distill_file="inranker_training_data_jul18.jsonl", max_length=2560
    # distill_file="beir_logits_1k.jsonl", max_length=2048
)

total_size = len(train_dataset)
validation_size = int(0.1 * total_size)
train_size = total_size - validation_size

dataset_dict = DatasetDict({"train": train_dataset})
split_datasets = dataset_dict["train"].train_test_split(test_size=validation_size, seed=42)

train_dataset = split_datasets["train"]
validation_dataset = split_datasets["test"]

def write_dataset_to_jsonl(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Reconstruct the original format
            output = {
                "query": item['query'],
                "contents": item['text'],
                "true_logit": item['label'][1],
                "false_logit": item['label'][0]
            }
            json.dump(output, f)
            f.write('\n')

# Write train dataset
write_dataset_to_jsonl(train_dataset, "inranker_training_data_jul18_train.jsonl")

# Write validation dataset
write_dataset_to_jsonl(validation_dataset, "inranker_training_data_jul18_validation.jsonl")

# trainer.train(train_dataset=train_dataset, validation_dataset=validation_dataset)
