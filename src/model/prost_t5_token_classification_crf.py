# Partially adapted from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L1716

from transformers import (
    T5EncoderModel,
    T5Config,
    modeling_outputs,
)

from src.model.crf import ConditionalRandomField

import torch
import torch.nn as nn

import src.config as model_config
from src.utils import truncate_prost_t5_emebddings_for_crf


class ProstT5EncoderModelTokenClassificationCRF(T5EncoderModel):
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
            out_features=self.custom_num_labels,
            device=self.device,
        )

        self.crf = ConditionalRandomField(
            num_tags=self.custom_num_labels,
            constraints=model_config.allowed_transitions_encoded,
            )
        
        self.post_init()

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
        print('encoder_outputs...')
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs["last_hidden_state"]
        
        print('truncate')
        input_ids, attention_mask, sequence_output, labels = truncate_prost_t5_emebddings_for_crf(input_ids, attention_mask, sequence_output, labels)

        sequence_output = self.custom_dropout(sequence_output)

        logits = self.custom_classifier(sequence_output)

        loss = None
        decoded_tags = None
        
        print('crf')
        if labels is not None:
            print('log')
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
            print('viterbi')
            decoded_tags = self.crf.viterbi_tags(
                logits=logits,
                mask=attention_mask.type(torch.uint8)
            )
        
        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        print('done')
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits if decoded_tags is None else decoded_tags,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
