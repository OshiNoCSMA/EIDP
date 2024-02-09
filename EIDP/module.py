import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import squareform, pdist

def gelu(x):
    """
    Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415

    Reference from: ICLRec: https://github.com/salesforce/ICLRec
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def elu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

ACTIVATION_FUNCTION = {
    "gelu": gelu,
    "relu": F.relu,
    "elu": elu
}

class LayerNorm(nn.Module):
    """
    Construct a layernorm module in the TF style (epsilon inside the square root).
    Reference from: ICLRec: https://github.com/salesforce/ICLRec
    """
    def __init__(self, hidden_size, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BatchNorm(nn.Module):
    '''
    take the input as the shape of (B, L, D), different from nn.BatchNorm1d (N, C, L)
    Reference from: https://github.com/Antinomy20001/BatchNorm_Pytorch_Experiment/blob/master/BatchNorm.py
    '''
    def __init__(self, hidden_dims):
        super(BatchNorm, self).__init__()
        self.hidden_dims = hidden_dims
        self.eps = 1e-5
        self.momentum = 0.1

        # hyper parameters
        self.gamma = nn.Parameter(torch.Tensor(self.hidden_dims), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.hidden_dims), requires_grad=True)

        # moving average
        self.moving_mean = torch.zeros(self.hidden_dims)
        self.moving_var = torch.ones(self.hidden_dims)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.moving_var)
        nn.init.zeros_(self.moving_mean)

    def batch_norm(self, x, gamma, beta, moving_mean, moving_var,
                   is_training=True, eps=1e-5, momentum=0.9):
        assert x.shape[-1] == self.hidden_dims
        mu = torch.mean(x, dim=(0, 1), keepdim=True) # (d, )
        var = torch.std(x, dim=(0, 1), unbiased=False) # (d, )
        if is_training:
            x_hat = (x - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mu
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        out = gamma * x_hat + beta

        return out, moving_mean, moving_var

    def forward(self, x):
        '''
        :param x: expected (b, L, d)
        '''
        self.moving_mean = self.moving_mean.to(x.device)
        self.moving_var = self.moving_var.to(x.device)

        bn_x, self.moving_mean, self.moving_var = self.batch_norm(x, self.gamma, self.beta,
                                                                  self.moving_mean, self.moving_var,
                                                                  self.training, self.eps, self.momentum)
        return bn_x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.num_heads = config['sal_heads']
        self.hidden_dims = config['hidden_dims']
        if self.hidden_dims % self.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_dims, self.num_heads)
            )
        self.head_size = int(self.hidden_dims / self.num_heads)
        self.reconstruct_size = self.num_heads * self.head_size

        self.Q = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.K = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.V = nn.Linear(self.hidden_dims, self.reconstruct_size)

        self.dropout = config['sal_dropout']
        self.attn_dropout_layer = nn.Dropout(self.dropout)
        self.FCL = nn.Linear(self.reconstruct_size, self.hidden_dims)
        self.Q_layer_norm = LayerNorm(self.hidden_dims)
        self.final_layer_norm = LayerNorm(self.hidden_dims)
        self.dropout_layer = nn.Dropout(self.dropout)

    def split_head(self, input):
        split_tensor_shape = input.size()[:-1] + (self.num_heads, self.head_size)
        input = input.view(*split_tensor_shape)
        # (b, h, L, d/h)
        return input.permute(0, 2, 1, 3)

    def forward(self, input, attn_mask):
        query = self.Q(self.Q_layer_norm(input))
        key = self.K(input)
        value = self.V(input)

        mh_q = self.split_head(query)
        mh_k = self.split_head(key)
        mh_v = self.split_head(value)

        # (b, h, L, d/h) * (b, h, d/h, L) --> (b, h, L, L)
        attn_scores = torch.matmul(mh_q, mh_k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_size)

        # (b, h, L, L) + (b, 1, L, L) --> (b, h, L, L)
        attn_scores += attn_mask

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout_layer(attn_probs)

        # (b, h, L, L) * (b, h, L, d/h) --> (b, h, L, d/h)
        mha_output = torch.matmul(attn_probs, mh_v)

        # (b, h, L, d/h) --> (b, L, h, d/h)
        mha_output = mha_output.permute(0, 2, 1, 3).contiguous()
        reconstruct_tensor_shape = mha_output.size()[:-2] + (self.reconstruct_size, )
        # (b, L, d)
        mha_output = mha_output.view(*reconstruct_tensor_shape)

        mha_output = self.FCL(mha_output)
        output = self.final_layer_norm(mha_output + input)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dims = config['hidden_dims']
        self.conv1 = nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=(1,))
        self.dropout1 = nn.Dropout(config['dropout'])
        if isinstance(config['ffn_acti'], str):
            self.FFN_acti_fn = ACTIVATION_FUNCTION[config['ffn_acti']]
        else:
            self.FFN_acti_fn = config['ffn_acti']
        self.conv2 = nn.Conv1d(self.hidden_dims, self.hidden_dims, kernel_size=(1,))
        self.dropout2 = nn.Dropout(config['dropout'])

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.FFN_acti_fn(
            self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.attention_layer = SelfAttention(config)
        self.feedforward_network = FeedForwardNetwork(config)

    def forward(self, seq_repr, attn_mask):
        output = self.attention_layer(seq_repr, attn_mask)
        output = self.feedforward_network(output)
        return output

class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        sal = SelfAttentionLayer(config)
        self.last_layernorm = LayerNorm(config['hidden_dims'])
        self.SAL = nn.ModuleList([copy.deepcopy(sal) for _ in
                                  range(config['sals'])])

    def forward(self, repr, attn_mask, store=True):
        each_SAL_layer_output = []
        for sub_layer in self.SAL:
            repr = sub_layer(repr, attn_mask)
            if store:
                repr = self.last_layernorm(repr)
                each_SAL_layer_output.append(repr)
        if not store:
            repr = self.last_layernorm(repr)
            each_SAL_layer_output.append(repr)
        return each_SAL_layer_output

class ProbMask():
    def __init__(self, B, H, L, index, scores, padding_mask=None, device="cpu"):
        '''
        :param L: in fact, is Lq
        :param index: (B, H, c*lnLq)
        :param scores: (B, H, c*lnLq, Lk)
        :param padding_mask: (B, L)
        '''

        # (Lq, Lk)
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # (B, H, Lq, Lk)
        _mask_ex = ~_mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # (B, H, α*lnLq, Lk)
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)

        # (B, H, α*lnLq, Lk)
        self._mask = indicator.view(scores.shape).to(device)

        if padding_mask is not None:
            # (B, H, L)
            _padding_mask = padding_mask.unsqueeze(-2).expand(B, H, L)
            # (B, H, α*lnLq)
            padding_indicator = _padding_mask[torch.arange(B)[:, None, None],
                                              torch.arange(H)[None, :, None], index].to(device)

            # (B, H, α*lnLq, Lk) * (B, H, α*lnLq, 1) --> (B, H, α*lnLq, Lk)
            self._mask = self._mask * padding_indicator.unsqueeze(-1)

    @property
    def mask(self):
        return self._mask.bool()

class ProbSparseAttention(SelfAttention):
    '''
    Implement based on the https://github.com/zhouhaoyi/Informer2020
    '''
    def __init__(self, config):
        super(ProbSparseAttention, self).__init__(config=config)

        # since the num of heads has been modified,
        # all related variables need to be redefined
        self.num_heads = config['lmsal_heads']
        if self.hidden_dims % self.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_dims, self.num_heads)
            )
        self.head_size = int(self.hidden_dims / self.num_heads)
        self.reconstruct_size = self.num_heads * self.head_size

        self.Q = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.K = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.V = nn.Linear(self.hidden_dims, self.reconstruct_size)
        self.FCL = nn.Linear(self.reconstruct_size, self.hidden_dims)

        self.dropout = config['lmsal_dropout']
        self.attn_dropout_layer = nn.Dropout(self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.alpha = config['alpha']
        self.with_mask = True
        self.eps = 1e-12

    def _prob_QK(self, Q, K, sample_k, n_top): # sample_k: α*lnLk; n_top: α*lnLq

        B, H, Lq, D = Q.shape
        _, _, Lk, _ = K.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, Lq, Lk, D)
        # sampled K index corresponds to Q
        index_sample = torch.randint(Lk, (Lq, sample_k))
        # (B, H, Lq, α*lnLk, D)
        K_sample = K_expand[:, :, torch.arange(Lq).unsqueeze(1), index_sample, :]
        # (B, H, Lq, 1, D) * (B, H, Lq, D, α*lnLk) --> (B, H, Lq, α*lnLk)
        # QK dot-product with the sampled keys
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Topk queries with sparsity measurement described in <Informer>
        # (B, H, Lq)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), Lk)
        # (B, H, α*lnLq)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q * K
        # (B, H, α*lnLq, D)
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]

        # (B, H, α*lnLq, D) * (B, H, D, Lk) --> (B, H, α*lnLq, Lk)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_values(self, V, Lq):
        B, H, Lv, D = V.shape

        if not self.with_mask:
            # (B, H, D/H)
            V_avg = V.mean(dim=-2)

            # (B, H, Lq, D/H)
            V_avg_for_padding_Q = V_avg.unsqueeze(-2).expand(B, H, Lq, D).clone()
        else:
            assert (Lq == Lv) # self-attention only
            # (B, H, Lv, D/H)
            V_avg = V.cumsum(dim=-2)

            # (B, H, Lq, D/H)
            V_avg_for_padding_Q = V_avg.clone()

        return V_avg_for_padding_Q

    def _update_values(self, V_avg, V, scores, index, Lq, padding_mask=None):
        '''
        Lk = Lv, Dq = Dk
        :param V_avg: (B, H, Lq, D/H)
        :param V: (B, H, Lv, D/H)
        :param scores: (B, H, c*lnLq, Lk)
        :param index: (B, H, c*lnLq)
        :param padding_mask: (B, L), only for self-attention
        '''

        B, H, Lv, D = V.shape
        if self.with_mask:
            # (B, H, α*lnLq, Lk)
            attn_mask = ProbMask(B, H, Lq, index, scores, padding_mask, device=V.device)
            scores.masked_fill_(~attn_mask.mask, -1e5)
            V_avg = V_avg / (padding_mask.cumsum(dim=-1).unsqueeze(1).unsqueeze(-1).expand(B, H, Lq, 1) + self.eps)

        # (B, H, α*lnLq, Lk)
        attn_scores = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # Lk = Lv; dq = dk, usually
        # (B, H, α*lnLq, Lk) * (B, H, Lv, D) --> (B, H, α*lnLq, D)
        V_avg[torch.arange(B)[:, None, None],
          torch.arange(H)[None, :, None],
          index, :] = torch.matmul(attn_scores, V).type_as(V_avg)

        # (B, H, Lq, D/H)
        return V_avg

    def forward(self, inputs, padding_mask=None):
        # suppose to be (B, L, d)
        (queries, keys, values) = inputs

        B, Lq, D = queries.shape
        _, Lk, _ = keys.shape
        H = self.num_heads

        proj_queries = self.Q(self.Q_layer_norm(queries))
        proj_keys = self.K(keys)
        proj_values = self.V(values)

        # (b, h, L, d/h)
        mh_q = self.split_head(proj_queries)
        mh_k = self.split_head(proj_keys)
        mh_v = self.split_head(proj_values)

        # α * ln(L_k)
        U_part = self.alpha * np.ceil(np.log(Lk)).astype('int').item()
        # α * ln(L_q)
        u = self.alpha * np.ceil(np.log(Lq)).astype('int').item()

        U_part = U_part if U_part < Lk else Lk
        u = u if u < Lq else Lq

        # (B, H, α*lnLq, Lk), (B, H, α*lnLq)
        top_queries, idx = self._prob_QK(mh_q, mh_k, sample_k=U_part, n_top=u)
        top_queries = top_queries * (1. / math.sqrt(D))

        # prepared the averaged values for padding with topk queries Q_K weighted sum
        # (B, H, Lq, D/H)
        values_for_padding = self._get_initial_values(mh_v, Lq)
        # update the averaged values
        # (B, H, Lq, D/H)
        output = self._update_values(values_for_padding,
                                     mh_v, top_queries, idx, Lq, padding_mask)

        # (B, H, Lq, D/H) -> (B, Lq, H, D/H)
        output = output.transpose(2, 1).contiguous()
        # (B, Lq, H, D/H) -> (B, Lq, D)
        output = output.view(B, Lq, -1)

        output = self.FCL(output)
        return self.final_layer_norm(output)

class LightMSAL(nn.Module):
    def __init__(self, config):
        super(LightMSAL, self).__init__()
        self.psa_layer = ProbSparseAttention(config)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, seq_repr, padding_mask):
        output = self.psa_layer((seq_repr, seq_repr, seq_repr), padding_mask)
        output = self.ffn(output)
        return output

class LightMSAB(nn.Module):
    def __init__(self, config):
        super(LightMSAB, self).__init__()
        light_msal = LightMSAL(config)
        self.last_layernorm = LayerNorm(config['hidden_dims'])
        self.LMSAL = nn.ModuleList([copy.deepcopy(light_msal) for _ in
                                  range(config['lmsals'])])

    def forward(self, repr, padding_mask, store=True):
        each_LMSAL_layer_output = []
        for sub_layer in self.LMSAL:
            repr = sub_layer(repr, padding_mask)
            if store:
                repr = self.last_layernorm(repr)
                each_LMSAL_layer_output.append(repr)
        if not store:
            repr = self.last_layernorm(repr)
            each_LMSAL_layer_output.append(repr)
        return each_LMSAL_layer_output

class DualCoupledBehaviorAttention(SelfAttention):
    def __init__(self, config):
        super(DualCoupledBehaviorAttention, self).__init__(config=config)

        self.user_num = config['user_num']
        self.behavior_types = config['behavior_types']
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        # |u| * bt+1 * 1
        self.user_bt_transition = nn.ModuleList([nn.Embedding(self.behavior_types + 1, 1)
                                                 for _ in range(self.user_num + 1)])

        self.Q_ib = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.K_ib = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.V_ib = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.Q_ib_layer_norm = LayerNorm(self.hidden_dims)

        self.Q_pb = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.K_pb = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.V_pb = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.Q_pb_layer_norm = LayerNorm(self.hidden_dims)

        self.FCL = nn.Linear(self.hidden_dims, self.hidden_dims)

        self.dropout = config['dcba_dropout']
        self.attn_dropout_layer = nn.Dropout(self.dropout)
        self.dropout_layer = nn.Dropout(self.dropout)

    def _generate_bs_hamming_dist_mat(self, u, bs_seq):
        '''bs_seq: (b, L, bt)'''
        batch_bs_hdm = []
        batch_size, l, bt = bs_seq.shape

        for b in range(batch_size):
            # (L, bt) --> (L, L)
            dist_idx = torch.LongTensor(squareform(pdist(bs_seq[b,:,:].cpu().numpy(), metric='hamming')) *
                                        self.behavior_types).to(self.device)
            u_dist = self.user_bt_transition[u[b]]
            bs_hamming_score = u_dist(dist_idx) # (L, L)
            batch_bs_hdm.append(bs_hamming_score.squeeze(-1))

        # (b, L, L)
        return torch.stack(batch_bs_hdm, dim=0)

    # incorporate behavior info.
    def forward(self, inputs, attn_mask):
        [u, seq_logits, item_emb, pos_emb, bs_seq, bs_emb] = inputs
        attn_mask = attn_mask.squeeze(1)

        ### item-behavior side
        # (b, L, d)
        ib_query = self.Q_ib(self.Q_ib_layer_norm(item_emb + bs_emb))
        ib_key = self.K_ib(item_emb + bs_emb)
        ib_value = self.V_ib(item_emb + bs_emb)

        # (b, L, d) * (b, d, L) --> (b, L, L)
        ib_attn_scores = torch.matmul(ib_query, ib_key.transpose(-1, -2))
        ib_attn_scores = ib_attn_scores / math.sqrt(self.hidden_dims)

        # (b, L, L)
        bs_hamming_attn = self._generate_bs_hamming_dist_mat(u, bs_seq) / math.sqrt(self.hidden_dims)

        # (b, L, L) + (b, L, L) --> (b, L, L)
        ib_side_attn_scores = nn.Softmax(dim=-1)(ib_attn_scores + attn_mask)
        # (b, L, L) + (b, L, L) --> (b, L, L)
        bs_hamming_attn = nn.Softmax(dim=-1)(bs_hamming_attn + attn_mask)

        ib_side_attn_scores = ib_side_attn_scores + bs_hamming_attn
        ib_side_attn_scores = self.attn_dropout_layer(ib_side_attn_scores)

        # (b, L, L) * (b, L, d) --> (b, L, d)
        ib_side_output = torch.matmul(ib_side_attn_scores, ib_value)

        ### position-behavior side
        # (b, L, d)
        pb_query = self.Q_pb(self.Q_pb_layer_norm(pos_emb + bs_emb))
        pb_key = self.K_pb(pos_emb + bs_emb)
        pb_value = self.V_pb(pos_emb + bs_emb)

        # (b, L, d) * (b, d, L) --> (b, L, L)
        pb_attn_scores = torch.matmul(pb_query, pb_key.transpose(-1, -2))
        pb_attn_scores = pb_attn_scores / math.sqrt(self.hidden_dims)

        # (b, L, L) + (b, L, L) --> (b, L, L)
        pb_side_attn_scores = nn.Softmax(dim=-1)(pb_attn_scores + attn_mask)

        pb_side_attn_scores = pb_side_attn_scores + bs_hamming_attn
        pb_side_attn_scores = self.attn_dropout_layer(pb_side_attn_scores)

        # (b, L, L) * (b, L, d) --> (b, L, d)
        pb_side_output = torch.matmul(pb_side_attn_scores, pb_value)

        mha_output = ib_side_output + pb_side_output

        # last fully-connected layer need or not?
        mha_output = self.FCL(mha_output)
        # mha_output = self.dropout_layer(mha_output)
        output = self.final_layer_norm(mha_output + seq_logits + bs_emb)

        return output


class CrossBehaviorAttention(nn.Module):
    def __init__(self, config):
        super(CrossBehaviorAttention, self).__init__()
        self.hidden_dims = config['hidden_dims']

        self.query_weight = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.key_weight = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.value_weight = nn.Linear(self.hidden_dims, self.hidden_dims)

        self.dense = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.layernorm = LayerNorm(self.hidden_dims)
        self.dropout_layer = nn.Dropout(config['cba_dropout'])

    def forward(self, query, key, value=None):
        '''
        :param query: bs embedding at the next time step in CBAF (\gamma^{\ell+1}),
        but can be sth. else instead, (b, L, d)
        :param key: seq logits of multiple behaivor-specific seq, (b, L, bt, d)
        :param value: the same as key, (b, L, bt, d)
        :return:
        '''

        if value is None:
            value = key.clone()

        # (b, L, d)
        Q = self.query_weight(query)
        # (b, L, bt, d)
        K = self.key_weight(key)
        # (b, L, bt, d)
        V = self.value_weight(value)

        # (b, L, bt, d) * (b, L, d) --> (b, L, bt)
        attn_scores = torch.einsum('bltd,bld->blt', K, Q)
        attn_scores = attn_scores / math.sqrt(self.hidden_dims)

        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.dropout_layer(attn_probs)
        # (b, L, bt) * (b, L, bt, d) --> (b, L, d)
        behavior_fusion_logits = torch.einsum('blt,bltd->bld', attn_probs, V)

        output = self.dense(behavior_fusion_logits)
        # output = self.dropout_layer(output)
        output = self.layernorm(output + query)

        return output