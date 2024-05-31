from time import sleep
import torch
import torch.nn as nn
from einops import rearrange
import os
from torch import Tensor
from sympy import symbols, floor
from layers.StandardNorm import Normalize

class Twin(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 2,
        d_layer = 2,
        head =8
    ):
        super().__init__()

        # attributes
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # Transformer encoder 1
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        self.encoder_1 = nn.TransformerEncoder(self.encoder_layer_1, num_layers = self.e_layer).to(self.device)

        # LLM embedding encoder 2
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        self.encoder_2 = nn.TransformerEncoder(self.encoder_layer_2, num_layers = self.e_layer).to(self.device)

        # Cross attention
        self.cross_layer = nn.TransformerDecoderLayer(d_model = self.num_nodes, nhead = 1, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        self.cross = nn.TransformerDecoder(self.cross_layer, num_layers = 1).to(self.device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        self.e_to_node = nn.Linear(self.d_llm, self.num_nodes).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN
        input_data = self.normalize_layers(input_data, 'norm')

        input_data = input_data.permute(0,2,1) # [B, V, L]
        input_data = self.length_to_feature(input_data) # [B, V, C]

        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1) # [B, E, V]
        embeddings = embeddings.permute(0,2,1) # [B, V, E]

        # Encoder
        enc_out = self.encoder_1(input_data) # [B, V, C]
        enc_out = enc_out.permute(0,2,1) # [B, C, V]

        embeddings = self.encoder_2(embeddings) # [B, V, E]
        embeddings = embeddings.permute(0,2,1) # [B, E, V]

        # Cross-attention
        cross_out = self.cross(enc_out, embeddings) # Q X KV  [B, C, V]X[B, E, V] = [B, C, V]
        cross_out = cross_out.permute(0,2,1) # [B, V, C]

        # Decoder
        dec_out = self.decoder(cross_out, cross_out) # [B, V, C]

        # Projection
        dec_out = self.c_to_length(dec_out) # [B, V, L]
        dec_out = dec_out.permute(0,2,1) # [B, L, V]

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
