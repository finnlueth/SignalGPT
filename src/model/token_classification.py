# Partially adapted from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L1716

from transformers import (
    T5EncoderModel,
    T5PreTrainedModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
)

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

# from torchcrf import CRF
from src.model.crf import ConditionalRandomField

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config


class T5EncoderModelForTokenClassification(T5EncoderModel):
    def __init__(
        self,
        config: T5Config,
        custom_num_labels,
        custom_dropout_rate,
        use_crf=False,
    ):
        super().__init__(config)

        self.custom_num_labels = custom_num_labels
        self.custom_dropout_rate = custom_dropout_rate
        self.use_crf = use_crf

        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)
        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels,
            device=self.device,
        )

        if self.use_crf:
            self.crf = ConditionalRandomField(
                num_tags=self.custom_num_labels,
                constraints=src.config.allowed_transitions_encoded,
                )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs["last_hidden_state"]

        input_ids = input_ids[:, 1:-1]
        attention_mask = attention_mask[:, 1:-1]
        labels = labels[:, :-2]

        input_ids = input_ids * (labels != -1)
        attention_mask = attention_mask * (labels != -1)

        sequence_output = sequence_output[:, 1:-1, :]
        sequence_output = self.custom_dropout(sequence_output)

        logits = self.custom_classifier(sequence_output)

        # print('input_ids', input_ids.shape, input_ids)
        # print('attention_mask', attention_mask.shape, attention_mask)
        # print('labels', labels.shape, labels)
        # print("logits", logits.shape, logits)

        loss = None
        decoded_tags = None

        if self.use_crf:
            if labels is not None:
                logits_crf = logits
                labels_crf = labels
                attention_mask_crf = attention_mask

                log_likelihood = self.crf(
                    inputs=logits_crf,
                    tags=labels_crf,
                    mask=attention_mask_crf.type(torch.uint8),
                )
                loss = -(log_likelihood / 1000)
            else:
                decoded_tags = self.crf.viterbi_tags(
                    logits=logits,
                    mask=attention_mask.type(torch.uint8)
                )
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                labels = labels.to(self.device)

                loss = loss_fct(
                    logits.reshape(-1, self.custom_num_labels),
                    labels.reshape(-1)
                )
                if loss > 2:
                    print("logits", logits.reshape(-1, self.custom_num_labels))
                    print("labels", labels.reshape(-1))
                    print("loss", loss)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits if decoded_tags is None else decoded_tags,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
