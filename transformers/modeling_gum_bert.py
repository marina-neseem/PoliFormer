# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings
from .controllers import BlockSelect

from .modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, load_tf_weights_in_bert, gelu, gelu_new, swish
from .modeling_bert import BertEncoder, BertPooler, BertEmbeddings, BertSelfAttention, \
                            BertAttention, BertIntermediate, BertOutput, BertLayer, \
                            BertLayerNorm, BertPreTrainedModel, BertModel, BertForSequenceClassification

logger = logging.getLogger(__name__)

class GumBertEncoder(BertEncoder):
    def __init__(self, config):
        super(GumBertEncoder, self).__init__(config)
        # TODO: 128 is max_seq_len
        self.controller = BlockSelect(128, config.hidden_size, config.num_hidden_layers, augmented=config.augmented)
        self.config = config

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, \
                encoder_attention_mask=None, return_num_executed_layers=False, adaptive=False, hard_gumbel=True, metrics=None):
        all_hidden_states = ()
        all_attentions = ()

        temp = hidden_states.clone()
        choices = self.controller(temp, hard_gumbel=hard_gumbel, metrics=metrics)
        for i, layer_module in enumerate(self.layer):
            if adaptive:
                if not self.training:
                    assert choices.shape[0] == 1, "Batch size has to be 1 during inference"
                    if choices[0, i] == 0:
                        continue

                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

                    layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
                    hidden_states = layer_outputs[0]
                else:
                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)
                
                    _, P, E = hidden_states.shape
                    reshaped_mask = choices[:, i].unsqueeze(-1).unsqueeze(-1).repeat(1, P, E)
                    tmp = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)[0]
                    hidden_states = (tmp * reshaped_mask) + (hidden_states * (torch.ones_like(reshaped_mask)-reshaped_mask) ) 

            else:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
                hidden_states = layer_outputs[0]

            # TODO 
            # if self.output_attentions:
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        
        if return_num_executed_layers:
            return outputs, choices
        else:
            return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GumBertModel(BertModel):
    def __init__(self, config):
        super(GumBertModel, self).__init__(config)
        self.encoder = GumBertEncoder(config)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                return_num_executed_layers=False, hard_gumbel=True, adaptive=True, metrics=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       return_num_executed_layers=return_num_executed_layers,
                                       hard_gumbel=hard_gumbel, adaptive=adaptive,
                                       metrics=metrics)
        if return_num_executed_layers:
            encoder_outputs, num_activated_layers = encoder_outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        
        if return_num_executed_layers:
            return outputs, num_activated_layers  # sequence_output, pooled_output, (hidden_states), (attentions)
        else:
            return outputs

class GumBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(GumBertForSequenceClassification, self).__init__(config)
        self.bert = GumBertModel(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                do_pretrain_controller_train=False, do_controller_train=False, 
                do_pretrain_model=False, target_compute=1.0, metrics=None):
        
        if do_pretrain_controller_train and do_controller_train:
            assert("Both do_controller_train and do_pretrain_controller_train are set True, choose where you stand and rerun")
        
        if do_controller_train:
            hard_gumbel = True
            adaptive = True
        elif do_pretrain_controller_train:
            hard_gumbel = False
            adaptive = True
        elif do_pretrain_model:
            hard_gumbel = False
            adaptive = False
        else:
            # Eval
            hard_gumbel = True
            adaptive = True
        
        outputs, num_activated_layers = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            return_num_executed_layers=True,
                            hard_gumbel=hard_gumbel,
                            adaptive=adaptive,
                            metrics=metrics)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if do_pretrain_controller_train:
            loss_fct = L1Loss()
            loss = loss_fct(num_activated_layers, torch.ones_like(num_activated_layers))
            outputs = (loss,) + outputs
            return outputs # (num_activated_layers), logits, (hidden_states), (attentions)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            if not do_pretrain_model and self.training:
                num_activated_layers = torch.sum(num_activated_layers, dim=-1)/self.bert.config.num_hidden_layers
                loss2_fct = MSELoss()
                loss2 = loss2_fct(num_activated_layers, target_compute*torch.ones_like(num_activated_layers))
                outputs = (loss, loss2) + outputs
            else:
                outputs = (loss, torch.sum(num_activated_layers, dim=-1)) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions)

