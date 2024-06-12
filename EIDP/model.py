import os
from module import *

class BasicModel(nn.Module):
    '''
    put some public attributes and methods here...
    '''
    def __init__(self, config):
        super(BasicModel, self).__init__()

        self.config = config
        self.user_num = config['user_num']
        self.item_num = config['item_num']
        self.behavior_types = config['behavior_types']
        self.cuda_condition = config['cuda_condition']
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.maxlen = config['maxlen']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout']
        self.init_std = config['init_std']

        self.save_path = config['output_dir']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.no = config['no']

    def init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        else:
            pass

class Encoder(BasicModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config=config)
        self.L_MSAB = LightMSAB(config)
        self.apply(self.init_weights)

    def forward(self, inputs):
        '''
        inputs:
        :param input_logits: (b, L, d)
        :param padding_mask: (b, L)
        :return:
        '''
        (input_logits, padding_mask) = inputs

        padding_mask = padding_mask.to(dtype=next(self.parameters()).dtype)
        seq_repr = self.L_MSAB(input_logits, padding_mask)
        output = seq_repr[-1]
        # (b, L, d)
        return output

class PBS_TPE(BasicModel):
    def __init__(self, config):
        super(PBS_TPE, self).__init__(config=config)
        self.dcba = DualCoupledBehaviorAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.layer_norm = LayerNorm(self.hidden_dims)

        self.apply(self.init_weights)

    def forward(self, inputs):
        '''
        inputs:
        :param input_logits: (b, L, d)
        :param attn_mask: (b, 1, L, L)
        :return:
        '''
        (input_logits, attn_mask) = inputs

        attn_mask = attn_mask.to(dtype=next(self.parameters()).dtype)
        seq_repr = self.dcba(input_logits, attn_mask)
        seq_repr = self.ffn(seq_repr)
        output = self.layer_norm(seq_repr)

        # (b, L, d)
        return output

class CBAF(BasicModel):
    def __init__(self, config):
        super(CBAF, self).__init__(config=config)
        self.cba = CrossBehaviorAttention(config=config)
        self.ffn = FeedForwardNetwork(config)
        self.layernorm = LayerNorm(self.hidden_dims)

    def forward(self, query, key, value=None):
        agg_output = self.cba(query=query, key=key, value=value)
        agg_output = self.ffn(agg_output)
        agg_output = self.layernorm(agg_output)
        return agg_output

class EIDP(BasicModel):
    def __init__(self, config):
        super(EIDP, self).__init__(config=config)

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_dims, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_dims)

        self.user_factors = nn.Embedding(self.user_num + 1, self.behavior_types)
        self.behavior_emb = nn.Embedding(self.behavior_types + 1, self.hidden_dims)

        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.padding_mask = None
        self.attn_mask = None

        # Implicit Modeling Path (IMP)
        ## UB-FEEL
        self.scaling_factor = self.behavior_types  # scaled factor

        ## L-MSAB
        self.BehaSpec_Encoder = nn.ModuleList()
        for _ in range(self.behavior_types):
            bs_encoder = Encoder(config=config)
            self.BehaSpec_Encoder.append(bs_encoder)

        ## CBAF
        self.cbaf = CBAF(config=config)

        # Explicit Modeling Path (EMP)
        ## PBS-TPE
        self.pbs_tpe = PBS_TPE(config=config)

        self.model_name = str(self.__class__.__name__)
        self.save_path = config['output_dir']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        fname = f"{self.model_name}_model.dataset={config['dataset']}." \
                f"ll={config['lmsals']}.lh={config['lmsal_heads']}.cd={config['cba_dropout']}." \
                f"dd={config['dcba_dropout']}.d={config['dropout']}.alpha={config['alpha']}." \
                f"seed={config['seed']}.no={self.no}.pth"
        self.save_path = os.path.join(self.save_path, fname)
        self.apply(self.init_weights)

    def set2emb(self, u, bs_seqs):
        '''
        :param u: (b, )
        :param bs_seqs: (b, L, bt)
        '''

        # (b, ) --> (b, bt) --> (b, 1, bt)
        batch_u_factors = self.user_factors(u.to(self.device)).unsqueeze(-2)
        # (b, 1, bt) * (b, L, bt) --> (b, L, bt)
        bs_factors = nn.Softmax(dim=-1)(batch_u_factors * bs_seqs)
        # (b, L, bt) * (1, bt, d) --> (b * L * d)
        behavior_set_embedding = torch.matmul(bs_factors, self.behavior_emb.weight[1:, :].unsqueeze(0))

        # (b, L, d)
        return behavior_set_embedding

    def ImplicitEncoding(self, input_seq):
        '''
        Implicit Path Encoding: item embedding + position embedding
        :param input_seq: (b, L)
        :return seq_emb: (b, L, d)
        '''
        position_ids = torch.arange(self.maxlen, dtype=torch.long,
                                    device=input_seq.device).unsqueeze(0).expand_as(input_seq)
        item_embeddings = self.item_emb(input_seq)
        item_embeddings *= self.item_emb.embedding_dim ** 0.5
        position_embeddings = self.pos_emb(position_ids)
        seq_emb = item_embeddings + position_embeddings
        seq_emb = self.dropout_layer(seq_emb)

        # (b, L, d)
        return seq_emb

    def ExplicitEncoding(self, u, input_seq, input_bs):
        '''
        Explicit Path Encoding:
        item embedding + position embedding + behavior set embedding
        :param u: (b, )
        :param input_seq: (b, L)
        :param input_bs: (b, L, bt)
        '''
        position_ids = torch.arange(self.maxlen, dtype=torch.long,
                                    device=input_seq.device).unsqueeze(0).expand_as(input_seq)
        item_embeddings = self.item_emb(input_seq)
        item_embeddings *= self.item_emb.embedding_dim ** 0.5

        position_embeddings = self.pos_emb(position_ids)

        # (b, L, d)
        behavior_set_embedding = self.set2emb(u, input_bs)

        seq_emb = item_embeddings + position_embeddings + behavior_set_embedding
        seq_emb = self.dropout_layer(seq_emb)

        # (b, L, d)
        return seq_emb, (item_embeddings, position_embeddings, behavior_set_embedding)

    def _generate_mask(self, input_ids):
        # construct attention mask
        # (b, L)
        padding_mask = (input_ids > 0).long()
        origin_padding_mask = padding_mask.clone()

        # (b, L) --> (b, 1, 1, L)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask_shape = (1, self.maxlen, self.maxlen)  # (1, L, L)
        # upper triangular, torch.uint8
        attn_mask = torch.triu(torch.ones(attn_mask_shape), diagonal=1)
        # --> bool
        attn_mask = (attn_mask == 0).unsqueeze(1)
        attn_mask = attn_mask.long()

        if self.config['cuda_condition']:
            attn_mask = attn_mask.cuda()

        # (b, 1, 1, L) * (1, 1, L, L)  --> (b, 1, L, L)
        attn_mask = padding_mask * attn_mask
        # (b, 1, L, L)
        attn_mask = (1.0 - attn_mask) * -1e5

        return origin_padding_mask, attn_mask

    def UB_FEEL(self, u, input_seqs, bs_seqs, target_bs):
        # (b, bt)
        batch_u_factors = self.user_factors(u.to(self.device))
        # (b, L, bt)
        batch_u_factors = batch_u_factors.unsqueeze(-2).repeat(1, self.maxlen, 1)
        # (b, L, bt)
        bs_factors = nn.Softmax(dim=-1)(batch_u_factors * bs_seqs)

        # (b, L, bt) * (1, bt, d) --> (b, L, d)
        target_bs_embs = torch.matmul(target_bs,
                                      self.behavior_emb.weight[1:, :].unsqueeze(0))

        # (b, L, bt)
        one_tensor = torch.ones_like(bs_seqs)
        # 1 or bt
        scaled_mask = torch.where(bs_seqs == 0, one_tensor, self.scaling_factor)

        # (b, L, bt)
        emhancement_mask = bs_factors * scaled_mask

        # (b, L, d)
        seq_logits = self.ImplicitEncoding(input_seqs)

        return (target_bs_embs, emhancement_mask, seq_logits)

    def _imp_forward(self, u, input_seqs, bs_seqs, target_bs):
        '''
        b: batch size;
        L: maxlen of seq;
        bt: behavior types num
        
        :param u: (b,)
        :param input_seqs: (b, L)
        :param bs_seqs: (b, L, bt)
        :param target_bs: (b, L, bt)
        :return: output: (b, L, d)
        '''
        bt = self.behavior_types

        ## UB-FEEL
        (target_bs_embs, emhancement_mask, seq_logits) = self.UB_FEEL(u, input_seqs, bs_seqs, target_bs)

        all_behavior_view_repr = torch.Tensor([]).to(self.device)
        for b in range(bt):
            bs_ehm_mask = emhancement_mask[:, :, b]  # (b, L)

            bs_seq_embs = seq_logits.clone()
            bs_seq_embs *= bs_ehm_mask.unsqueeze(-1)

            view_repr = self.BehaSpec_Encoder[b]((bs_seq_embs,  # (b, L, d)
                                               self.padding_mask)) # (b, L)

            # (b, L, bt, d)
            all_behavior_view_repr = torch.cat((all_behavior_view_repr,
                                                view_repr.unsqueeze(-2)), dim=-2)

        X_imp = self.cbaf(target_bs_embs, all_behavior_view_repr, all_behavior_view_repr)

        # (b, L, d)
        return X_imp

    def _emp_forward(self, u, input_seqs, bs_seqs):
        '''
        b: batch size;
        L: maxlen of seq;
        bt: behavior types num
        
        :param u: (b,)
        :param input_seqs: (b, L)
        :param bs_seqs: (b, L, bt)
        :return: output: (b, L, d)
        '''
        # (b, L, d)
        seq_logits, (item_emb, pos_emb, bs_emb) = self.ExplicitEncoding(u, input_seqs, bs_seqs)

        # (b, L, d)
        X_emp = self.pbs_tpe(
            ([u, seq_logits, item_emb, pos_emb, bs_seqs, bs_emb],
             self.attn_mask)
        )

        return X_emp

    def forward(self, inputs):
        '''
        b: batch size;
        L: maxlen of seq;
        bt: behavior types num
        '''
        (u, input_seqs, bs_seqs, target_bs) = inputs
        self.padding_mask, self.attn_mask = self._generate_mask(input_seqs)

        X_emp = self._emp_forward(u, input_seqs, bs_seqs) # (b, L, d)
        X_imp = self._imp_forward(u, input_seqs, bs_seqs, target_bs) # (b, L, d)

        x = (X_emp + X_imp) / 2 # (b, L, d)
        return x

