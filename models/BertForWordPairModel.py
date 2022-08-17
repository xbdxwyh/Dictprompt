import torch

from transformers import (
    BertPreTrainedModel,
    BertModel
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

from .WordPairOutput import WordEmbdPairOutput
from typing import Optional, Tuple, Union

class BertForWordPairClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

        # Customized
        self.embedding_size = config.hidden_size
        self.linear_diff = torch.nn.Linear(config.hidden_size, 250, bias = True)
        self.linear_seperator = torch.nn.Linear(250, 2, bias = True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        word1_locs: Optional[torch.FloatTensor] = None,
        word2_locs: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_model = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embs = outputs_model[0]
        batch_size = word1_locs.shape[0]
        index_word1 = ([i for i in range(batch_size)],word1_locs)
        index_word2 = ([i for i in range(batch_size)],word2_locs)

        word1s = embs[index_word1]
        word2s = embs[index_word2]
        diff = word1s - word2s


        # Calculate outputs using activation
        layer1_results = self.activation(self.linear_diff(diff))
        logits = self.softmax(self.linear_seperator(layer1_results))
        loss = None
        #print(logits)
        #print(labels)
        if labels is not None:
            #  We want seperation like a SVM so use Hinge loss
            loss = self.loss(logits.view(-1, 2), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs_model.hidden_states,
            attentions=outputs_model.attentions,
        )


class BertForWordPairEmbeddings(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        word1_locs: Optional[torch.FloatTensor] = None,
        word2_locs: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_model = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embs = outputs_model[0]
        batch_size = word1_locs.shape[0]
        index_word1 = ([i for i in range(batch_size)],word1_locs)
        index_word2 = ([i for i in range(batch_size)],word2_locs)

        word1s = embs[index_word1]
        word2s = embs[index_word2]

        return WordEmbdPairOutput(
            embd_word1 = word1s,
            embd_word2 = word2s,
            hidden_states = embs
        )