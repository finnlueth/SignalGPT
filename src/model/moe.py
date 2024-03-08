def predict_model(
    sequence: str,
    tokenizer: T5Tokenizer,
    model: T5EncoderModelForTokenClassification,
    attention_mask=None,
    labels=None,
    device="cpu",
    viterbi_decoding=False,
):
    if tokenizer:
        tokenized_string = tokenizer.encode(
            sequence,
            padding="max_length",
            max_length=71,
            truncation=True,
            return_tensors="pt",
        ).to(device)
    # print(tokenized_string.shape)
    # print(attention_mask.shape)
    with torch.no_grad():
        output = model(
            input_ids=tokenized_string.to(device),
            labels=None if viterbi_decoding else labels,
            attention_mask=attention_mask,
        )
    return output


def translate_logits(logits, decoding, viterbi_decoding=False):
    # print('logits', logits, type(logits))
    if viterbi_decoding:
        return [decoding[x] for x in logits[0]]
    else:
        return [decoding[x] for x in logits.cpu().numpy().argmax(-1).tolist()[0]]


def moe_inference(
    sequence,
    tokenizer,
    model_gate,
    model_expert,
    labels=None,
    attention_mask=None,
    device="cpu",
    result_type=None,
    use_crf=False,
):
    if not result_type:
        print("Determining type...")
        gate_preds = src.model.predict_model(
            sequence=sequence,
            tokenizer=tokenizer,
            model=model_gate,
            labels=labels,
            attention_mask=attention_mask,
            device=device,
        )

        result_type = src.model.translate_logits(
            logits=gate_preds.logits.unsqueeze(0), decoding=src.config.type_decoding
        )[0]
        print("Found Type: ", result_type)

    try:
        assert result_type in src.config.type_encoding.keys()
    except Exception as e:
        print(e)
        print(f'Type "{result_type}" not found. Exiting...')

    tmp_custom_classifier = nn.Linear(
        in_features=model_expert.config.hidden_size,
        out_features=len(src.config.select_decoding_type[result_type]),
    )
    if use_crf:
        tmp_crf = CRF(
            num_tags=len(src.config.select_decoding_type[result_type]), batch_first=True
        )
        model_expert.custom_classifier.weight = tmp_custom_classifier.weight
        model_expert.custom_classifier.bias = tmp_custom_classifier.bias
        model_expert.crf.start_transitions = tmp_crf.start_transitions
        model_expert.crf.end_transitions = tmp_crf.end_transitions
        model_expert.crf.transitions = tmp_crf.transitions

    model_expert.set_adapter(result_type)

    pred_logits = src.model_new.predict_model(
        sequence=sequence,
        tokenizer=tokenizer,
        model=tokenizer,
        labels=labels,
        attention_mask=attention_mask,
        device=device,
        viterbi_decoding=use_crf,
    )

    pred_sequence = src.model_new.translate_logits(
        logits=pred_logits.logits,
        decoding=src.config.select_decoding_type[result_type],
        viterbi_decoding=use_crf,
    )

    model_expert.unload()

    return result_type, pred_sequence


def moe_train():
    pass
