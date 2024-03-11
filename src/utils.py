from pathlib import Path


def get_project_root_path() -> Path:
    return str(Path(__file__).parent.parent)


def truncate_prost_t5_emebddings_for_crf(input_ids, attention_mask, sequence_output, labels):
    input_ids = input_ids[:, 1:-1]
    attention_mask = attention_mask[:, 1:-1]
    labels = labels[:, :-2]

    input_ids = input_ids * (labels != -1)
    attention_mask = attention_mask * (labels != -1)

    sequence_output = sequence_output[:, 1:-1, :]

    return input_ids, attention_mask, sequence_output, labels
