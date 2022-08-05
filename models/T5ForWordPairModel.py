import torch
from torch import nn
import copy
from typing import Optional, Tuple, Union

from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5Stack
)

from transformers.models.t5.configuration_t5 import T5Config

from transformers.modeling_outputs import (
    BaseModelOutput,
    SequenceClassifierOutput
)

num_labels=2

class T5ForWordPairClassification(T5PreTrainedModel):
    def __init__(self,config: T5Config,num_labels=num_labels):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        # 自己定义的部分（学着BERT那块的）
        self.num_labels = num_labels
        
        # 自定义
        self.embedding_size = config.hidden_size
        self.linear_diff = torch.nn.Linear(config.hidden_size, 250, bias = True)
        self.linear_seperator = torch.nn.Linear(250, 2, bias = True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        word1_locs: Optional[torch.FloatTensor] = None,
        word2_locs: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        embs = encoder_outputs[0]
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
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )