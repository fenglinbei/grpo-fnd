import os
import json
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader

from src.datasets.schemas import Sample, LabelEnum
from src.datasets.sent_matcher import align_tokenized_to_raw_with_meta

LABEL2ID = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "half": 3,
    "mostly-true": 4,
    "true": 5,
}



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

def build_liar_datasets(train_file, val_file, test_file):
    train_dataset = build_dataset_from_liar(train_file)
    val_dataset = build_dataset_from_liar(val_file)
    test_dataset = build_dataset_from_liar(test_file)
    return train_dataset, val_dataset, test_dataset

def build_and_save_liar_datasets(train_file, val_file, test_file, output_dir):
    train_dataset, val_dataset, test_dataset = build_liar_datasets(train_file, val_file, test_file)
    save_dataset(train_dataset, f'{output_dir}/train.json')
    save_dataset(val_dataset, f'{output_dir}/val.json')
    save_dataset(test_dataset, f'{output_dir}/test.json')

def build_dataset_from_rawfc(split_dir):
    raw_data = []
    dataset: list[Sample] = []
    for filename in os.listdir(split_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(split_dir, filename)
            raw_data.append(load_json(file_path))
    for item in tqdm(raw_data, desc="Processing samples"):
        
        sample_id = int(item['event_id'])
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

def build_and_save_rawfc_datasets(train_dir, val_dir, test_dir, output_dir):
    train_dataset = build_dataset_from_rawfc(train_dir)
    val_dataset = build_dataset_from_rawfc(val_dir)
    test_dataset = build_dataset_from_rawfc(test_dir)
    save_dataset(train_dataset, f'{output_dir}/train.json')
    save_dataset(val_dataset, f'{output_dir}/val.json')
    save_dataset(test_dataset, f'{output_dir}/test.json')

def load_dataset(dataset_path: str) -> list[Sample]:
    dataset = load_json(dataset_path)
    return [Sample.model_validate(sample) for sample in dataset]
    
class VeracityJsonDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.data = load_dataset(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        item = self.data[idx]
        return item

if __name__ == "__main__":
    # test_data_dir = "data/raw/RAWFC/test"
    # test_data_file_list = os.listdir(test_data_dir)
    # for filename in test_data_file_list[:1]:
    #     test_file_exp = os.path.join(test_data_dir, filename)
    #     json_data = load_json(test_file_exp)
    #     print(list(json_data.keys()))
    #     # print(json_data['event_id'])
    #     print(list(json_data['reports'][0].keys()))
    #     print(json_data['reports'][0]['tokenized'][0])

    # build_and_save_liar_datasets(
    #     train_file="data/raw/LIAR-RAW/train.json",
    #     val_file="data/raw/LIAR-RAW/val.json",
    #     test_file="data/raw/LIAR-RAW/test.json",
    #     output_dir="data/processed/LIAR"
    # )

    # build_and_save_rawfc_datasets(
    #     train_dir="data/raw/RAWFC/train",
    #     val_dir="data/raw/RAWFC/val",
    #     test_dir="data/raw/RAWFC/test",
    #     output_dir="data/processed/RAWFC"
    # )
    pass
