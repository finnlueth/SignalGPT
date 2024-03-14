from src import config, utils
import requests
import pandas as pd

from transformers import (
    T5EncoderModel,
    T5PreTrainedModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
)

from datasets import Dataset, DatasetDict

import src.config


def download_file(url: str, name: str, path: str):
    """Download file from url and save to file_path
    Args:
        url (_type_): _description_
        file_path (_type_): _description_
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(path + name, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully to '{path}'.")
    else:
        print(f"Failed to download file to {path} .")


def download_all(path: str, urls: dict):
    """Download all files to path
    Args:
        path (_type_): _description_
    """
    for name, url in urls.items():
        download_file(url=url, path=path, name=name)


def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="UTF-8") as file:
        file = file.read()

    items = []
    file = file.splitlines()
    for index, line in enumerate(file):
        if line.startswith(">"):
            line = line[1:].split("|")
            data = {
                "Uniprot_AC": line[0],
                "Kingdom": line[1],
                "Type": line[2],
                "Partition_No": line[3],
                "Sequence": file[index + 1],
                "Label": file[index + 2],
            }
            items.append(pd.Series(data=data, index=data.keys()))
    return pd.DataFrame(items)


def process_old(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data.Sequence = df_data.Sequence.apply(lambda x: " ".join([*(x)]))
    df_data.Label = df_data.Label.apply(
        lambda x: [config.label_encoding[x] for x in [*(x)]]
    )
    # df_process_data['Mask'] = df_process_data.Annotation.apply(lambda x: [0 if item == -1 else 1 for item in x])
    df_data["Split"] = df_data["Partition_No"].map(int)
    # df_data['Split'] = df_data.Partition_No.apply(lambda x: "test" if x in ['4'] else "train")
    # df_data.drop(columns=['Partition_No', 'Uniprot_AC', 'Kingdom', 'Type'], inplace=True)
    return df_data


def process(df_data: pd.DataFrame) -> pd.DataFrame:
    df_data.Sequence = df_data.Sequence.apply(lambda x: "<AA2fold> " + " ".join([*(x)]))
    df_data["Partition_No"] = df_data["Partition_No"].map(int)
    # df_data.Label = df_data.Label.apply(lambda x: [config.label_encoding[x] for x in [*(x)]])
    # df_process_data['Mask'] = df_process_data.Annotation.apply(lambda x: [0 if item == -1 else 1 for item in x])
    # df_data['Split'] = df_data.Partition_No.apply(lambda x: "test" if x in ['4'] else "train")
    # df_data.drop(columns=['Partition_No', 'Uniprot_AC', 'Kingdom', 'Type'], inplace=True)
    return df_data


# def df_to_fasta(df, fasta_file):
#     with open(fasta_file, 'w') as file:
#         for index, row in df.iterrows():
#             header = f">{row['Uniprot_AC']}|{row['Kingdom']}|{row['Type']}|{row['Partition_No']}"
#             sequence = row['Sequence']
#             label = row['Label']
#             file.write(f"{header}\n{sequence}\n")
#             # file.write(f"{header}\n{sequence}\n{label}\n")


def df_to_dataset(
    tokenizer: T5Tokenizer, sequences: list, labels: list, encoder: dict
) -> Dataset:
    tokenized_sequences = tokenizer(
        sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024
    )
    dataset = Dataset.from_dict(tokenized_sequences)
    dataset = dataset.add_column(
        "labels", [encoder[x] for x in labels], new_fingerprint=None
    )
    return dataset


def create_datasets(
    splits: dict,
    tokenizer: T5Tokenizer,
    data: pd.DataFrame,
    annotations_name: list,
    dataset_size: int = None,
    sequence_type=None,
) -> DatasetDict:
    datasets = {}

    for split_name, split in splits.items():
        # print(split_name, split)
        if sequence_type and sequence_type != "ALL":
            data_split = data[data["Type"] == sequence_type]
        else:
            data_split = data
        if dataset_size:
            data_split = data_split[data_split.Partition_No.isin(split)]
            data_split = data_split.sample(
                n=min(data_split.shape[0], dataset_size * len(split)),
                random_state=1,
                replace=False
            )
        else:
            data_split = data_split[data_split.Partition_No.isin(split)]

        tokenized_sequences = tokenizer(
            data_split["Sequence"].to_list(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )
        dataset = Dataset.from_dict(tokenized_sequences)
        if "Label" in annotations_name:
            encoder = src.config.select_encoding_type[sequence_type]
            # print(data_split)
            dataset = dataset.add_column(
                "labels",
                [[encoder[y] for y in x] for x in data_split["Label"].to_list()],
                new_fingerprint=None,
            )
        if "Type" in annotations_name:
            encoder = src.config.type_encoding
            dataset = dataset.add_column(
                "type",
                [encoder[x] for x in data_split["Type"].to_list()],
                new_fingerprint=None,
            )
        datasets[split_name] = dataset

    # for x in datasets.values():
    #     x.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return DatasetDict(datasets)


# def create_datasets_head(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict) -> DatasetDict:
#     datasets = {}

#     for split_name, split in splits.items():
#         data_split = data[data.Partition_No.isin(split)].head(dataset_size * len(split) if dataset_size else dataset_size)
#         if
#         tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
#         dataset = Dataset.from_dict(tokenized_sequences)
#         if annotations_name == 'Label':
#             dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
#         if annotations_name == 'Type':
#             dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
#         datasets[split_name] = dataset
#     return DatasetDict(datasets)
