params = {x[0]: x[1] for x in t5_base_model.named_parameters()}
params['crf.modules_to_save.default._constraint_mask']


# print(*params.keys(), sep='\n')

aaa = torch.nn.Parameter(torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 1., 0., 0., 0., 0., 1.],
        [0., 0., 1., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]]), requires_grad=False)

t5_base_model.crf.original_module._constraint_mask, t5_base_model.crf.original_module.tensor_constraint_mask

t5_base_model.crf.modules_to_save.default._constraint_mask, t5_base_model.crf.modules_to_save.default.tensor_constraint_mask

# [x for x in t5_lora_model.custom_classifier.modules_to_save.default.named_parameters()]

# display(pd.Series([item for row in dataset_signalp['train']['labels'] for item in row]).value_counts())
# display(pd.Series([item for row in dataset_signalp['valid']['labels'] for item in row]).value_counts())
# display(pd.Series([item for row in dataset_signalp['test']['labels'] for item in row]).value_counts())

# src.model_new.make_confusion_matrix(
#     training_log['eval_confusion_matrix'].iloc[-1],
#     model_config.select_decoding_type[EXPERT])

# _ds_index = 3250
# _ds_index = 3250
# _ds_type = 'test'
# USE_CRF = False

# _input_ids_test = t5_tokenizer.decode(dataset_signalp[_ds_type][_ds_index]['input_ids'][:-1])
# _labels_test = torch.tensor([dataset_signalp[_ds_type][_ds_index]['labels'] + [-100]]).to(device)
# _attention_mask_test = torch.tensor([dataset_signalp[_ds_type][_ds_index]['attention_mask']]).to(device)

# _labels_test_decoded = [model_config.label_decoding[x] for x in _labels_test.tolist()[0][:-1]]

# print('Iput IDs:\t', _input_ids_test)
# print('Labels:\t\t', *_labels_test.tolist()[0])
# print('Labels Decoded:\t', *_labels_test_decoded)
# print('Attention Mask:\t', *_attention_mask_test.tolist()[0])
# print('----')

# preds = src.model_new.predict_model(
#     sequence=_input_ids_test,
#     tokenizer=t5_tokenizer,
#     model=t5_lora_model,
#     labels=_labels_test,
#     attention_mask=_attention_mask_test,
#     device=device,
#     viterbi_decoding=USE_CRF,
#     )

# _result = src.model_new.translate_logits(
#     logits=preds.logits,
#     viterbi_decoding=USE_CRF,
#     decoding=model_config.label_decoding
#     )

# print('Result: \t',* _result)



# index = 1
# print(len(dataset_signalp['train']['input_ids'][index]), dataset_signalp['train']['input_ids'][index])
# print(len(dataset_signalp['train']['labels'][index]), dataset_signalp['train']['labels'][index])
# print(len(dataset_signalp['train']['attention_mask'][index]), dataset_signalp['train']['attention_mask'][index])



# for x in range(3):
#     print(len(dataset_signalp['valid'][x]['labels']))
#     print(*dataset_signalp['valid'][x]['labels'])


crf = src.model.ConditionalRandomField(
                num_tags=7,
                constraints=src.config.allowed_transitions_encoded,
                )


# print('input_ids', input_ids.shape, input_ids)
# print('attention_mask', attention_mask.shape, attention_mask)
# print('labels', labels.shape, labels)
# print("logits", logits.shape, logits)



# sequence_output = self.custom_dropout(sequence_output)

# print('sequence_output.requires_grad', sequence_output.requires_grad)
# print(mean_sequence_output.shape)
# print(mean_sequence_output)
# print('sequence_output', sequence_output[:, 0, :].shape, sequence_output[:, 0, :])
# print()
# print('sequence_output', sequence_output.shape, sequence_output)

# print('-------------')
# print('encoder_outputs', encoder_outputs.hidden_states)


# print('logits', logits_crf.shape, logits_crf.dtype, logits_crf, logits_crf.min(), logits_crf.max())
# print('labels', labels_crf.shape, labels_crf.dtype, labels_crf, labels_crf.min(), labels_crf.max())
# print('attention_mask', attention_mask_crf.shape, attention_mask_crf.dtype, attention_mask_crf)


# model_config.select_encoding_type.keys()
# df_data = src.data.process(src.data.parse_file(ROOT + '/data/raw/' + model_config.dataset_name))
# df_data['Label'].apply(lambda x: len(x)).describe()