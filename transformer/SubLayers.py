import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention

#Implementing external stimuli aware attention
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # changing the following to encode endo and exogeneous events, 1 end , 2, 3,  ex
        # self.event_type = event_type

        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_en_ex_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ex_en_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ex_ex_qs = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_uniform_(self.w_qs.weight)
#         nn.init.xavier_uniform_(self.w_en_ex_qs.weight)
        nn.init.xavier_uniform_(self.w_ex_en_qs.weight)
#         nn.init.xavier_uniform_(self.w_ex_ex_qs.weight)

        # changed above
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, event_type, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        event_type_dict = Constants.event_type_dict_moodle
        event_type = event_type.cpu().numpy()
        event_type = np.vectorize(event_type_dict.get)(event_type)
        event_type = torch.tensor(event_type)

        batch_size, event_type_dim = event_type.shape

        event_type_prev = event_type.reshape(batch_size, event_type_dim, 1).repeat(1, 1, event_type_dim)
        event_type_present = event_type.reshape(batch_size, 1, event_type_dim).repeat(1, event_type_dim, 1)
        en_en_flag = torch.logical_and(event_type_prev == True, event_type_present == True)
        ex_en_flag = torch.logical_and(event_type_prev == False, event_type_present == True)

        q_en_en = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        q_ex_en = self.w_ex_en_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q_en_en = q_en_en.transpose(1, 2)
        q_ex_en = q_ex_en.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        
        
        #Implementing external stimuli aware attention like LUKE
        q_en_en_attn = self.attention(q_en_en, k, v, mask=mask)
        q_ex_en_attn = self.attention(q_ex_en, k, v, mask=None)


        en_en_flag = einops.repeat(en_en_flag, 'b h w -> (repeat b) h w', repeat=n_head)
        en_en_flag = en_en_flag.reshape(sz_b, n_head, event_type_dim, event_type_dim)


        ex_en_flag = einops.repeat(ex_en_flag, 'b h w -> (repeat b) h w', repeat=n_head)
        ex_en_flag = ex_en_flag.reshape(sz_b, n_head, event_type_dim, event_type_dim)


        en_en_flag = en_en_flag.to(torch.device('cuda'))
        ex_en_flag = ex_en_flag.to(torch.device('cuda'))



        final_q_luke_attn = en_en_flag * q_en_en_attn + ex_en_flag * q_ex_en_attn

        attn = self.dropout(F.softmax(final_q_luke_attn, dim=-1))
        output = torch.matmul(attn, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
