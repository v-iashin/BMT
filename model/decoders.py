import torch
import torch.nn as nn

from model.blocks import (BridgeConnection, LayerStack,
                          PositionwiseFeedForward, ResidualConnection, clone)
from model.multihead_attention import MultiheadedAttention


class DecoderLayer(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H)
        self.enc_att = MultiheadedAttention(d_model, d_model, d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)

    def forward(self, x, memory, src_mask, trg_mask):
        ''' 
        in:
            x, memory - (B, S, D) src_mask (B, 1, S) trg_mask (B, S, S)
        out:
            x, memory - (B, S, D)
        '''
        # a comment regarding the motivation of the lambda function # please see the EncoderLayer
        def sublayer0(x): return self.self_att(x, x, x, trg_mask)
        def sublayer1(x): return self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward

        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)

        return x


class BiModalDecoderLayer(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model_C, d_model, dout_p, H, d_ff_C):
        super(BiModalDecoderLayer, self).__init__()
        # self attention
        self.res_layer_self_att = ResidualConnection(d_model_C, dout_p)
        self.self_att = MultiheadedAttention(d_model_C, d_model_C, d_model_C, H, dout_p, d_model)
        # encoder attention
        self.res_layer_enc_att_A = ResidualConnection(d_model_C, dout_p)
        self.res_layer_enc_att_V = ResidualConnection(d_model_C, dout_p)
        self.enc_att_A = MultiheadedAttention(d_model_C, d_model_A, d_model_A, H, dout_p, d_model)
        self.enc_att_V = MultiheadedAttention(d_model_C, d_model_V, d_model_V, H, dout_p, d_model)
        # bridge
        self.bridge = BridgeConnection(2*d_model_C, d_model_C, dout_p)
        # feed forward residual
        self.res_layer_ff = ResidualConnection(d_model_C, dout_p)
        self.feed_forward = PositionwiseFeedForward(d_model_C, d_ff_C, dout_p)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc) 
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc) 
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        C, memory = x
        Av, Va = memory

        # Define sublayers
        # a comment regarding the motivation of the lambda function please see the EncoderLayer
        def sublayer_self_att(C): return self.self_att(C, C, C, masks['C_mask'])
        def sublayer_enc_att_A(C): return self.enc_att_A(C, Av, Av, masks['A_mask'])
        def sublayer_enc_att_V(C): return self.enc_att_V(C, Va, Va, masks['V_mask'])
        sublayer_feed_forward = self.feed_forward

        # 1. Self Attention
        # (B, Sc, Dc)
        C = self.res_layer_self_att(C, sublayer_self_att)

        # 2. Encoder-Decoder Attention
        # (B, Sc, Dc) each
        Ca = self.res_layer_enc_att_A(C, sublayer_enc_att_A)
        Cv = self.res_layer_enc_att_V(C, sublayer_enc_att_V)
        # (B, Sc, 2*Dc)
        C = torch.cat([Ca, Cv], dim=-1)
        # bridge: (B, Sc, Dc) <- (B, Sc, 2*Dc)
        C = self.bridge(C)

        # 3. Feed-Forward
        # (B, Sc, Dc) <- (B, Sc, Dc)
        C = self.res_layer_ff(C, sublayer_feed_forward)

        return C, memory


class Decoder(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(d_model, dout_p, H, d_ff), N)

    def forward(self, x, memory, src_mask, trg_mask):
        '''
        in:
            x (B, S, D) src_mask (B, 1, S) trg_mask (B, S, S)
        out:
            (B, S, d_model)
        '''
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)

        return x


class BiModelDecoder(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model_C, d_model, dout_p, H, d_ff_C, N):
        super(BiModelDecoder, self).__init__()
        layer = BiModalDecoderLayer(
            d_model_A, d_model_V, d_model_C, d_model, dout_p, H, d_ff_C
        )
        self.decoder = LayerStack(layer, N)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        # x is (C, memory)
        C, memory = self.decoder(x, masks)

        return C
