import json
from inranker import T5Ranker
from tqdm import tqdm
import pickle

model = T5Ranker(model_name_or_path="castorini/monot5-3b-msmarco-10k", fp8=True)
# load data from inranker_training_data_jul18.jsonl
data = []
with open("inranker_training_data_jul18.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))


def write_labeled_data_to_file(labeled_data, file_path):
    with open(file_path, "w") as f:
        for d in labeled_data:
            f.write(json.dumps(d) + "\n")


current_query = None
docs_batch = []
labeled_data = []
for d in tqdm(data):
    if d["query"] != current_query:
        # new query, start new batch to label, label previous batch
        if docs_batch:
            # label batch
            scores, logits_batch = model.get_scores(
                query=current_query, docs=docs_batch, return_logits=True
            )
            for doc, logits in zip(docs_batch, logits_batch):
                labeled_data.append(
                    {
                        "query": current_query,
                        "contents": doc,
                        "false_logit": logits[0],
                        "true_logit": logits[1],
                    }
                )
            write_labeled_data_to_file(
                labeled_data, "inranker_training_data_monoT5_jul24.jsonl"
            )
        # reset batch
        docs_batch = [d["contents"]]
        current_query = d["query"]
    else:
        # same query, add to batch
        docs_batch.append(d["contents"])

# label last batch
scores, logits_batch = model.get_scores(
    query=current_query, docs=docs_batch, return_logits=True
)

for doc, logits in zip(docs_batch, logits_batch):
    labeled_data.append(
        {
            "query": current_query,
            "contents": doc,
            "false_logit": logits[0],
            "true_logit": logits[1],
        }
    )

write_labeled_data_to_file(labeled_data, "inranker_training_data_monoT5_jul24.jsonl")
