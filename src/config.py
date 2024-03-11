base_model_name = 'Rostlab/prot_t5_xl_uniref50'
# base_model_name = 'Rostlab/ProstT5'
model_version = 0

# Debug
VERBOSE = True

# Model
dropout_rate = 0.1
use_crf = True
seed = 42
selected_expert = 'ALL'

# Data
data_model_version = 6
dataset_name = f'{data_model_version}_SignalP_{data_model_version}.0_Training_set.fasta'
raw_dataset_path = '/data/raw/'
processed_dataset_path = '/data/processed/'
dataset_size = 3
# dataset_size = None
data_name = f'sp{data_model_version}_unprocessed'

# Training
steps = 30
lr = 1e-4
batch_size = 16
num_epochs = 1
# num_epochs = 2
save_steps = 9999999
logging_steps = 1
# eval_steps = steps*5
eval_steps = 300
# metric = compute_metrics_fast

model_name = 'linear_model_v4'

splits = {
    'train': [0, 1],
    'valid': [2],
    # 'test': [2]
}

# splits_all = {'all': [0, 1, 2, 3, 4]}

# Encodings
label_encoding = {"I": 0, "L": 1, "M": 2, "O": 3, "S": 4, "T": 5, "P": 6}
label_decoding = dict(zip(label_encoding.values(), label_encoding.keys()))

label_encoding_NO_SP = {"I": 0, "M": 1, "O": 2}
label_decoding_NO_SP = dict(zip(label_encoding_NO_SP.values(), label_encoding_NO_SP.keys()))

label_encoding_SP = {"I": 0, "M": 1, "O": 2, "S": 3}
label_decoding_SP = dict(zip(label_encoding_SP.values(), label_encoding_SP.keys()))

label_encoding_LIPO = {"I": 0, "L": 1, "M": 2, "O": 3}
label_decoding_LIPO = dict(zip(label_encoding_LIPO.values(), label_encoding_LIPO.keys()))

label_encoding_TAT = {"O": 0, "T": 1}
label_decoding_TAT = dict(zip(label_encoding_TAT.values(), label_encoding_TAT.keys()))

label_encoding_PILIN = {"O": 0, "M": 1, "P": 2}
label_decoding_PILIN = dict(zip(label_encoding_TAT.values(), label_encoding_TAT.keys()))


type_encoding = {'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3, 'PILIN': 3, 'TATLIPO': 4}
type_decoding = dict(zip(type_encoding.values(), type_encoding.keys()))

select_encodings = {'Label': label_encoding, 'Type': type_encoding}

select_encoding_type = {'ALL': label_encoding, 'NO_SP': label_encoding_NO_SP, 'SP': label_encoding_SP, 'LIPO': label_encoding_LIPO, 'TAT': label_encoding_TAT}
select_decoding_type = {'ALL': label_decoding, 'NO_SP': label_decoding_NO_SP, 'SP': label_decoding_SP, 'LIPO': label_decoding_LIPO, 'TAT': label_decoding_TAT}

# Allowed transitions (see 00.1_data_analysis)
allowed_transitions = [('I', 'I'), ('I', 'M'), ('L', 'L'), ('L', 'O'), ('M', 'I'), ('M', 'M'), ('M', 'O'), ('O', 'M'), ('O', 'O'), ('P', 'O'), ('P', 'P'), ('S', 'O'), ('S', 'S'), ('T', 'O'), ('T', 'T'), ('<START>', 'I'), ('<START>', 'L'), ('<START>', 'O'), ('<START>', 'P'), ('<START>', 'S'), ('<START>', 'T'), ('I', '<END>'), ('M', '<END>'), ('O', '<END>')]
allowed_transitions_encoded = [(label_encoding[t[0]] if t[0] != '<START>' else max(label_encoding.values())+1,
                                label_encoding[t[1]] if t[1] != '<END>' else max(label_encoding.values())+2) for t in allowed_transitions]

# Dataset URLs
urls = {
    '6_SignalP_6.0_Training_set.fasta':     'https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta',
    '6_SignalP_5.0_Benchmark_set.fasta':    'https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/benchmark_set_sp5.fasta',
    '5_SignalP_5.0_Training_set.fasta':     'https://services.healthtech.dtu.dk/services/SignalP-5.0/train_set.fasta',
    '5_SignalP_5.0_Benchmark_set.fasta':    'https://services.healthtech.dtu.dk/services/SignalP-5.0/benchmark_set.fasta',
}
