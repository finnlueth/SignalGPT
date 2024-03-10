from transformers import (
    T5EncoderModel,
    T5Config,
    modeling_outputs,
)

from torchcrf import CRF

import torch
import torch.nn as nn


class ProtT5EncoderModelTokenClassificationCRF(T5EncoderModel):
    def __init__(
        self,
        config: T5Config,
        custom_num_labels,
        custom_dropout_rate,
    ):
        super().__init__(config)

        self.custom_num_labels = custom_num_labels
        self.custom_dropout_rate = custom_dropout_rate

        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)
        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )

        self.crf = CRF(
            num_tags=self.custom_num_labels,
            batch_first=True
        )

    # Adapted from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L1716
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs['last_hidden_state']

        sequence_output = self.custom_dropout(sequence_output)
        logits = self.custom_classifier(sequence_output)

        loss = None
        decoded_tags = None

        if labels is not None:
            logits_crf = logits[:, :-1, :]
            labels_crf = labels[:, :-1]
            # fix mask for shorter sequences
            attention_mask_crf = attention_mask[:, :-1]
            attention_mask_crf[labels_crf == -100] = 0
            labels_crf[labels_crf == -100] = 0
            log_likelihood = self.crf(
                emissions=logits_crf,
                tags=labels_crf,
                mask=attention_mask_crf.type(torch.uint8)
            )
            loss = -log_likelihood/1000
        else:
            decoded_tags = self.crf.decode(logits, mask=attention_mask.type(torch.uint8))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits if decoded_tags is None else decoded_tags,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )