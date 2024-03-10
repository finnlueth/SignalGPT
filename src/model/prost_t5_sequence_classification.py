from transformers import (
    T5EncoderModel,
    T5Config,
    modeling_outputs,
)

from torch.nn import CrossEntropyLoss

import torch.nn as nn


# Adapted from https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/t5/modeling_t5.py#L775
class ProstT5EncoderModelForSequenceClassification(T5EncoderModel):
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

        self.custom_classifier_in = nn.Linear(
            in_features=self.config.hidden_size, out_features=self.config.hidden_size
        )
        self.custom_classifier_out = nn.Linear(
            in_features=self.config.hidden_size, out_features=self.custom_num_labels
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # position_ids=None,
        # head_mask=None,
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
        
        # todo: truncate

        logits = sequence_output.mean(dim=1)
        logits = self.custom_dropout(logits)
        logits = self.custom_classifier_in(logits)
        logits = logits.tanh()
        logits = self.custom_dropout(logits)
        logits = self.custom_classifier_out(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs['last_hidden_state'],
            # attentions=encoder_outputs.attentions,
        )
