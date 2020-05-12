from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerStack(nn.Module):

    def __init__(self, layer, N):
        super(LayerStack, self).__init__()
        self.layers = clone(layer, N)

    def forward(self, x, masks):
        for layer in self.layers:
            x = layer(x, masks)
        return x

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VocabularyEmbedder(nn.Module):

    def __init__(self, voc_size, emb_dim):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        # replaced if pretrained weights are used
        self.embedder = nn.Embedding(voc_size, emb_dim)

    def forward(self, x):  # x - tokens (B, seq_len)
        x = self.embedder(x)
        x = x * np.sqrt(self.emb_dim)

        return x  # (B, seq_len, d_model)

    def init_word_embeddings(self, weight_matrix, emb_weights_req_grad=True):
        if weight_matrix is None:
            print('Training word embeddings from scratch')
        else:
            pretrained_voc_size, pretrained_emb_dim = weight_matrix.shape
            if self.emb_dim == pretrained_emb_dim:
                self.embedder = self.embedder.from_pretrained(weight_matrix)
                self.embedder.weight.requires_grad = emb_weights_req_grad
                print('Glove emb of the same size as d_model_caps')
            else:
                self.embedder = nn.Sequential(
                    nn.Embedding(self.voc_size, pretrained_emb_dim).from_pretrained(weight_matrix),
                    nn.Linear(pretrained_emb_dim, self.emb_dim),
                    nn.ReLU()
                )
                self.embedder[0].weight.requires_grad = emb_weights_req_grad


class FeatureEmbedder(nn.Module):

    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        self.embedder = nn.Linear(d_feat, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        # (B, S, d_model_m) <- (B, S, D_original_feat_dim)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        x = self.activation(x)

        # (B, S, d_model_m)
        return x


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        # torch.cuda.FloatTensor torch.FloatTensor
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        # same as input
        return x


class Transpose(nn.Module):
    """
        LayerNorm expects (B, S, D) but receives (B, D, S)
        Conv1d expects (B, D, S) but receives (B, S, D)
    """

    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class ResidualConnection(nn.Module):

    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):  
        # x (B, S, D)
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)

        return x + res


class BridgeConnection(nn.Module):

    def __init__(self, in_dim, out_dim, dout_p):
        super(BridgeConnection, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return self.activation(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dout_p = dout_p
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x):
        '''In, Out: (B, S, D)'''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
