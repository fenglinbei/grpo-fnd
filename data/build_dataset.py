import sys
sys.path.append('.')
import json
from enum import Enum
from pydantic import BaseModel
from typing import List
from tqdm import tqdm
from loguru import logger

from data.sent_matcher import align_tokenized_to_raw_with_meta

LABEL2ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "half": 3,
    "mostly-true": 4,
    "true": 5,
}

class LabelEnum(int, Enum):
    PANTS_FIRE = 0
    FALSE = 1
    BARELY_TRUE = 2
    HALF_TRUE = 3
    MOSTLY_TRUE = 4
    TRUE = 5

class Sample(BaseModel):
    id: int
    claim: str
    label: LabelEnum
    explanation: str
    evidence: List[str]

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def build_dataset_from_liar(file_path):
    raw_data = load_json(file_path)
    dataset = []
    for item in tqdm(raw_data, desc="Processing samples"):
        sample_id = item['event_id'][:-5] if item['event_id'].endswith(".json") else item['event_id']
        evidence_list = []
        for r in item['reports']:
            align_result = align_tokenized_to_raw_with_meta(raw_text=r["content"], tokenized=r['tokenized'])
            for s in align_result:
                if s["is_evidence"]:
                    evidence_list.append(s["raw_sent"])
        unique_evidence = list(set(evidence_list))
        sample = Sample(
            id=sample_id,
            claim=item['claim'],
            label=LabelEnum(LABEL2ID[item['label']]),
            explanation=item['explain'],
            evidence=unique_evidence
        )
        dataset.append(sample)
    return dataset

def save_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        json.dump([sample.model_dump() for sample in dataset], f, indent=2)

if __name__ == "__main__":
    train_file_path = 'data/raw/LIAR-RAW/train.json'
    # val_file_path = 'data/raw/LIAR-RAW/val.json'
    # test_file_path = 'data/raw/LIAR-RAW/test.json'
    # print(list(dataset[0].keys()))
    # for i in range(1):
    #     print(list(dataset[i]['reports'][0].keys()))
    #     print(json.dumps(dataset[i]['reports'][0], indent=2))

    train_dataset = build_dataset_from_liar(train_file_path)
    # val_dataset = build_dataset_from_liar(val_file_path)
    # test_dataset = build_dataset_from_liar(test_file_path)
    # print(dataset[0])
    save_dataset(train_dataset, 'data/processed/LIAR-RAW/train.json')
    # save_dataset(val_dataset, 'data/processed/LIAR-RAW/val.json')
    # save_dataset(test_dataset, 'data/processed/LIAR-RAW/test.json')