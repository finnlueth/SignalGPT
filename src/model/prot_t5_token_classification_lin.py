from transformers import (
    T5EncoderModel,
    T5Config,
    modeling_outputs,
)

from torch.nn import (
    CrossEntropyLoss,
)

import torch.nn as nn


class ProtT5EncoderModelTokenClassificationLinear(T5EncoderModel):
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

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(self.device)

            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )