"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import bisect
import collections
import json
import operator
import time

import numpy as np
import pandas as pd
import torch

import common
import made
import transformer

from torch.nn import functional as F

def in_between(data, val) -> bool:
    assert len(val) == 2
    lrange, rrange = val
    return np.greater_equal(data, lrange) & np.less_equal(data, rrange)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '[]': in_between
}


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s


def FillInUnqueriedColumns(table, columns, operators, vals):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """
    ncols = len(table.columns)
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    return cs, os, vs


class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.shortcircuit = shortcircuit

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r
        # self.num_samples = 1
        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros((self.num_samples, len(self.model.input_bins)), device=device))

        self.dom_sizes = [c.DistributionSize() for c in self.table.columns]
        self.dom_sizes = np.cumsum(self.dom_sizes)

        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print('Setting masked_weight in MADE, do not retrain!')
        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)
            self.inp = self.traced_encode_input(self.kZeros)

            # For transformer, need to flatten [num cols, d_model].
            self.inp = self.inp.view(self.num_samples, -1)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None):
        ncols = len(columns)
        logits = self.init_logits
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        for i in range(ncols):
            natural_idx = ordering[i]

            # Column i.
            op = operators[natural_idx]
            if op is not None:
                # There exists a filter.
                valid_i = OPS[op](columns[natural_idx].all_distinct_values,
                                  vals[natural_idx]).astype(np.float32,
                                                            copy=False)
            else:
                valid_i = np.ones(columns[natural_idx].DistributionSize(),)
                # continue

            # This line triggers a host -> gpu copy, showing up as a
            # hotspot in cprofile.
            valid_i_list[i] = torch.as_tensor(valid_i, dtype=torch.float32, device=self.device)
        print()
        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = i if ordering is None else ordering[i]
                if operators[natural_idx] is None and natural_idx != ncols - 1:
                    if natural_idx == 0:
                        self.model.EncodeInput(
                            None,
                            natural_col=0,
                            out=inp[:, :self.model.
                                    input_bins_encoded_cumsum[0]])
                    else:
                        l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                                 1]
                        r = self.model.input_bins_encoded_cumsum[natural_idx]
                        self.model.EncodeInput(None,
                                               natural_col=natural_idx,
                                               out=inp[:, l:r])

        # Actual progressive sampling.  
        # jacobi version.
        num_jacobi_window = 3
        
        ids_to_input = None # samples used in Naru
        last_logits = None
        ids_keep_unconverged_tokens = None
        ids_accept_tokens = None
        unfinished_sequences = torch.ones(self.num_samples, dtype=torch.long, device=self.device)
        num_keep_unconverged_token = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        # matric_last_unconverged_logits = torch.zeros((self.num_samples, self.model.input_bins_encoded_cumsum[num_jacobi_window-1]), device=self.device)
        
        has_unfinished_sequences = True
        
        prefix_token_sampler = SpeculativeSampler(valid_i_list, num_jacobi_window, self.model.input_bins_encoded_cumsum, ncols)
        
        while has_unfinished_sequences:
            
            # ids_to_input: x ~ p_{\theta}(x_{i}|x^{j-1}_{1:i-1})
            
            ids_to_input, mask_forward_tokens, last_logits = self.prepare_inputs_for_generation_jacobi(
                ids_to_input, 
                last_logits,
                valid_i_list,
                ncols,
                num_jacobi_window,
                num_keep_unconverged_token,
                ids_accept_tokens,
            ) # [B, ncol], [B, ncol]
            
            logits = self.model(ids_to_input)
            
            # get next token of each unconverged col
            # accept ids don't need to be sampled
            # between a forward pass and a reject sampling in an iteration : len(unconverged tokens) = num_jacobi_window
            # |-- accepted tokens --|-- unconverged tokens --|-- residual tokens --|
            
            cur_unconverged_probs = [] # list(p_{\theta}(x_{i}|x^{j}_{1:i-1}))
            last_unconverged_probs = [] # list(p_{\theta}(x_{i}|x^{j-1}_{1:i-1}))
            unconverged_probs_summed = []
            unconverged_tokens = [] # list(x ~ p_{\theta}(x_{i}|x^{j}_{1:i-1}))
            min_unconverged_id = min(ids_accept_tokens) + 1 if ids_accept_tokens is not None else 0
            max_unconverged_id = min(max(ids_accept_tokens) + 1 + num_jacobi_window, ncols) if ids_accept_tokens is not None else min(num_jacobi_window, ncols)
            print(f'unconvergedleft bound : {min_unconverged_id}, unconverged id right bound : {max_unconverged_id-1}')
            for i in range(min_unconverged_id, max_unconverged_id):
                if i == 0:
                    l = 0
                    r = self.model.input_bins_encoded_cumsum[0]
                else:
                    l = self.model.input_bins_encoded_cumsum[i - 1]
                    r = self.model.input_bins_encoded_cumsum[i]
                    
                probs_i = torch.softmax(logits[:, l:r], 1)
                last_probs_i = last_logits[:, l:r]
                    
                valid_i = valid_i_list[i]
                
                if valid_i is not None:
                    probs_i *= valid_i
                    
                probs_i_summed = probs_i.sum(1)
                # print(f'{columns[natural_idx]} probs_i_summed : {probs_i_summed.shape}')
                unconverged_probs_summed.append(probs_i_summed)

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                
                if self.shortcircuit and operators[natural_idx] is None:
                    sample_ids = None
                else:
                    sample_ids = torch.multinomial(probs_i, 1, replacement=True)  # [B, 1]
                    ids_to_input[:, i] = sample_ids.view(-1)
                
                unconverged_tokens.append(sample_ids)    
    
                batch_indices = torch.arange(self.num_samples, device=probs_i.device).unsqueeze(1)
                cur_unconverged_probs.append(probs_i[batch_indices, sample_ids])
                last_unconverged_probs.append(last_probs_i[batch_indices, sample_ids])
            
            col_ids = torch.arange(ncols, dtype=torch.int32, device=self.device).unsqueeze(0).expand(self.num_samples, -1) # [B, ncol]
            
            mask_accept_tokens = (col_ids <= ids_accept_tokens.unsqueeze(1)).int() if ids_accept_tokens is not None else 0 # [B, ncol]
            # mask_met_tokens = (col_ids <= (ids_accept_tokens if ids_accept_tokens is not None else 0 + num_jacobi_window).unsqueeze(1)).int() # [B, ncol]
            # mask_sampled_tokens = mask_met_tokens - mask_accept_tokens # [B, ncol]
            mask_unconverged_tokens = mask_forward_tokens - mask_accept_tokens # [B, ncol]
            # print(f'mask_forward_tokens : {mask_forward_tokens} mask_accept_tokens : {mask_accept_tokens}')
            # matric_unconverged_tokens = torch.cat(unconverged_tokens, dim=1) * mask_unconverged_tokens[:, min_unconverged_id:max_unconverged_id] # [B, max_unconverged_id - min_unconverged_id]
            matric_cur_unconverged_probs = torch.cat(cur_unconverged_probs, dim=1) # [B, max_unconverged_id - min_unconverged_id]
            matric_last_unconverged_probs = torch.cat(last_unconverged_probs, dim=1) # [B, max_unconverged_id - min_unconverged_id], contains new rand tokens' probs which are not in the last iteration
            # matric_cur_unconverged_logits = logits[:, self.model.input_bins_encoded_cumsum[min_unconverged_id]:self.model.input_bins_encoded_cumsum[max_unconverged_id]] # [B, subset_unconverged(nin)]
            # matric_sampled_tokens = torch.cat(unconverged_tokens, dim=1) * mask_sampled_tokens[:, min_unconverged_id:max_unconverged_id] # [B, max_unconverged_id - min_unconverged_id]
            
            # Speculative Decoding
            num_new_accept_tokens, ids_accept_tokens, ids_to_input \
                = self.prefix_matching_next_tokens(
                    min_unconverged_id, max_unconverged_id, prefix_token_sampler, 
                    matric_last_unconverged_probs, matric_cur_unconverged_probs, 
                    last_logits, logits, 
                    mask_unconverged_tokens, ids_to_input, ids_accept_tokens, 
                    unconverged_probs_summed
                ) # [B,], [B,], [B, nin]
            
            num_keep_unconverged_token = num_jacobi_window - num_new_accept_tokens # [B,]
            # ids_keep_unconverged_tokens += num_keep_unconverged_token # [B,]
            # print(f'num_new_accept_tokens : {num_new_accept_tokens}')
            # early stop 
            unfinished_sequences = (ids_accept_tokens < ncols - 1).int()
            has_unfinished_sequences = (unfinished_sequences.max() != 0)
            last_logits = logits.clone()
            del logits
        print(f'accept_probs: {prefix_token_sampler.accept_probs}') 
        masked_probs = prefix_token_sampler.accept_probs
           
        # Doing this convoluted scheme because m_p[0] is a scalar, and
        # we want the corret shape to broadcast.
        p = masked_probs[1]
        for ls in masked_probs[2:]:
            p *= ls
        p *= masked_probs[0]

        return p.mean().item()

    def Query(self, columns, operators, vals):
        # Massages queries into natural order.
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)

        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)
        # print('Ordering: {}'.format(ordering))

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
        inv_ordering = [None] * len(columns)
        for natural_idx in range(len(columns)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            inp_buf = self.inp.zero_()
            # Fast (?) path.
            if num_orderings == 1:
                ordering = orderings[0]
                self.OnStart()
                p = self._sample_n(
                    self.num_samples,
                    ordering if isinstance(
                        self.model, transformer.Transformer) else inv_ordering,
                    columns,
                    operators,
                    vals,
                    inp=inp_buf)
                self.OnEnd()
                return np.ceil(p * self.cardinality).astype(dtype=np.int32,
                                                            copy=False)

            # Num orderings > 1.
            ps = []
            self.OnStart()
            for ordering in orderings:
                p_scalar = self._sample_n(self.num_samples // num_orderings,
                                          ordering, columns, operators, vals)
                ps.append(p_scalar)
            self.OnEnd()
            return np.ceil(np.mean(ps) * self.cardinality).astype(
                dtype=np.int32, copy=False)

    def prepare_inputs_for_generation_jacobi(
        self,
        ids_to_input,
        last_logits,
        valid_i_list,
        num_col,
        num_jacobo_window,
        num_keep_unconverged_token,
        ids_accept_tokens,
    ):
        # goal: (accepted tokens, unconverged tokens) + (new initialized random tokens)
        # num_col: total number of columns
        # ids_accept_tokens: indices of sequences that are accepted
        #
        # num_jacobo_window = num_keep_unconverged + num_new_init
        # len(ids_to_input) = num_col
        # |-- accepted tokens --|-- unconverged tokens --|-- new initialized random tokens --|-- residual tokens --|
        # |<-      fixed      ->|<-   reject sampled   ->|<-             random            ->|<- residual_tokens ->|
        #
        # return: 
        # ids_to_input: [B, ncol], model input
        # mask_forward_tokens: [B, ncol], indicates which tokens are Speculatived in this iteration
        # last_logits: [B, nin], logits of last iteration
        
        rand_tokens = None # [B, max_last_rand_id - min_last_rand_id]
                    
        if ids_to_input is not None and last_logits is not None:
            min_last_rand_id = min(ids_accept_tokens + num_keep_unconverged_token) + 1
            max_last_rand_id = max(ids_accept_tokens) + num_jacobo_window + 1
            # print(f'ids_accept_tokens: {ids_accept_tokens} last_rand_l: {min_last_rand_id}, last_rand_r: {max_last_rand_id}')
            for i in range(min_last_rand_id, min(max_last_rand_id, num_col)):
                l = self.model.input_bins_encoded_cumsum[i - 1]
                r = self.model.input_bins_encoded_cumsum[i]
                last_probs_i = torch.softmax(last_logits[:, l:r], 1)
                valid_i = valid_i_list[i]
                
                if valid_i is not None:
                    last_probs_i *= valid_i
                
                last_probs_i_summed = last_probs_i.sum(1)
                # print(f'{columns[natural_idx]} probs_i_summed : {probs_i_summed.shape}')

                # If some paths have vanished (~0 prob), assign some nonzero
                # mass to the whole row so that multinomial() doesn't complain.
                paths_vanished = (last_probs_i_summed <= 0).view(-1, 1)
                last_probs_i = last_probs_i.masked_fill_(paths_vanished, 1.0)
                last_logits[:, l:r] = last_probs_i
                
                if rand_tokens is None:
                    rand_tokens = torch.multinomial(last_probs_i, 1, True).view(-1, 1).int()
                else:
                    rand_tokens = torch.cat([rand_tokens, torch.multinomial(last_probs_i, 1, True).view(-1, 1).int()], dim=1)
            
            col_ids = torch.arange(num_col).unsqueeze(0).expand(self.num_samples, -1).int().cuda() # [B, ncol]
            forward_ids = (ids_accept_tokens + num_jacobo_window).unsqueeze(1) # [B,]
            mask_forward_tokens = (col_ids <= forward_ids).int() # [B, ncol]
            ids_to_input[:, min_last_rand_id:max_last_rand_id] = \
                ids_to_input[:, min_last_rand_id:max_last_rand_id] * mask_forward_tokens[:, min_last_rand_id:max_last_rand_id] + \
                rand_tokens * (1 - mask_forward_tokens[:, min_last_rand_id:max_last_rand_id]) if rand_tokens is not None else 0
            
        else:
            ids_accept_tokens = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
            min_last_rand_id = 0
            max_last_rand_id = num_jacobo_window
            
            for i in range(num_col):
                if ids_to_input is None and last_logits is None:
                    ids_to_input = torch.multinomial(valid_i_list[i], self.num_samples, True).view(-1, 1)
                    last_logits = torch.softmax(valid_i_list[i], dim=-1).unsqueeze(0).expand(self.num_samples, -1)
                else:
                    ids_to_input = torch.cat([ids_to_input, torch.multinomial(valid_i_list[i], self.num_samples, True).view(-1, 1)], dim=1)  
                    last_logits = torch.cat([last_logits, torch.softmax(valid_i_list[i], dim=-1).unsqueeze(0).expand(self.num_samples, -1)], dim=1)  
        
            col_ids = torch.arange(num_col).unsqueeze(0).expand(self.num_samples, -1) # [B, ncol]
            mask_forward_tokens = (col_ids < num_jacobo_window).int() # [B, ncol]
            
        # print(f'before speculative ids_to_input: {ids_to_input}')
        return ids_to_input.int(), mask_forward_tokens, last_logits
        
    def prefix_matching_next_tokens(
        self,
        min_unconverged_id, max_unconverged_id,
        prefix_token_sampler,
        draft_probs,
        cur_unconverged_probs,
        last_logits,
        logits,
        mask_unconverged_tokens,
        ids_to_input, ids_accept_tokens,
        unconverged_probs_summed
    ):
        # draft_probs / unconverged_probs: contain random tokens' probs
        mask_anti_unconverged_tokens = 1 - mask_unconverged_tokens[:, min_unconverged_id:max_unconverged_id]
        draft_probs[mask_anti_unconverged_tokens.int() == 1] = 1
        cur_unconverged_probs[mask_anti_unconverged_tokens.int() == 1] = 1
        
        num_new_accept_tokens, ids_accept_tokens, ids_to_input = prefix_token_sampler(
            ids_to_input, ids_accept_tokens,
            min_unconverged_id, max_unconverged_id,
            draft_prob=draft_probs,
            advanced_prob=cur_unconverged_probs,
            last_logits=last_logits,
            logits=logits,
            mask_unconverged_tokens=mask_unconverged_tokens
        )
        
        # stat accept cols
        num_new_accept_col = num_new_accept_tokens.min().item()
    
        for i in range(num_new_accept_col):
            prefix_token_sampler.accept_probs.append(unconverged_probs_summed[i])
            
        return num_new_accept_tokens, ids_accept_tokens, ids_to_input


class SpeculativeSampler:
    def __init__(self, valid_i_list, num_jacobi_window, input_bins_encoded_cumsum, ncols):

        self.valid_i_list = valid_i_list
        self.num_jacobi_window = num_jacobi_window
        self.input_bins_encoded_cumsum = input_bins_encoded_cumsum
        self.ncols = ncols
        self.accept_probs = []
        self.shortcircuit = False
    
    def get_reject_sampling_logits(self, token_advanced_prob, token_draft_prob):
        pos_delta_logits = (
            token_advanced_prob - token_draft_prob
        ).clamp(min=0).log()
        return pos_delta_logits

    def reject_sampling_col(self, last_logits, logits, i):
        # last_logits: [subset(B), bin_i], logits: [subset(B), bin_i]
        # return: resampled_tokens: [subset(B),]
        pos_delta_logits = self.get_reject_sampling_logits(logits, last_logits)
        resampled_logits = F.softmax(pos_delta_logits, dim=-1)
        valid_i = self.valid_i_list[i]
        # print(f'last_logits: {last_logits.shape} logits: {logits.shape}')
        # print(f'resampled_logits: {resampled_logits.shape} valid_i: {valid_i.shape if valid_i is not None else None}')
        if valid_i is not None:
            resampled_logits *= valid_i

        paths_vanished = (resampled_logits.sum(dim=-1) <= 0).view(-1, 1)
        resampled_logits = resampled_logits.masked_fill_(paths_vanished, 1.0)
        
        if self.shortcircuit and valid_i is None:
            resampled_tokens = None
        else:
            samples_i = torch.multinomial(
                resampled_logits, num_samples=1,
                replacement=True)  
            resampled_tokens = samples_i.view(-1, 1) # [subset(B), 1]
            
        return resampled_tokens
    
    def __call__(self, ids_to_input, ids_accept_tokens, 
                 min_unconverged_id, max_unconverged_id, 
                 draft_prob, advanced_prob, 
                 last_logits, logits, 
                 mask_unconverged_tokens): 
        # draft_tokens: [B, L], draft_prob: [B, L], advanced_prob: [B, L]
        # mask_unconverged_tokens: [B, ncol], only contains unconverged tokens, exclude random tokens and accept tokens
        # L = min(ids_accept_tokens) + 1 : max(ids_accept_tokens) + num_jacobi_window

        # reinitalize self.reject_sampling_relative_ids
        print(f'draft_prob : {draft_prob} , advanced_prob : {advanced_prob}')
        mask_unconverged_splite = mask_unconverged_tokens[:, min_unconverged_id:max_unconverged_id].cuda() # [B, L]
        # print(f'mask_unconverged_tokens : {mask_unconverged_splite}')
        rs = torch.rand(advanced_prob.shape, device=advanced_prob.device) # [B, L]
        
        mask_accept_criterions = (rs < (advanced_prob / draft_prob).clamp(max=1)).int() * mask_unconverged_splite # [B, L]
        print(f'mask_accept_criterions : {mask_accept_criterions}')
        # The accepted tokens for each sample can be identified as the first contiguous subsequence consisting entirely of 1s 
        # within the longest common prefix per row of mask_unconverged_splite and mask_accept_criterions.
        diff = (mask_accept_criterions ^ mask_unconverged_splite) * mask_unconverged_splite # [B, L]
        # print(f'diff : {diff}')
        max_vals, first_diff_pos = diff.max(dim=1) # [B,], [B,]
        num_common_prefixes = torch.where(max_vals == 0, torch.tensor(diff.shape[1], device=max_vals.device), first_diff_pos).int() 
        print(f'num_common_prefixes : {num_common_prefixes}')
        # Adjust the out-of-range part
        
        if ids_accept_tokens is None:
            # the first token is always accepted
            ids_accept_tokens = - torch.ones(ids_to_input.shape[0], device=ids_to_input.device).int()
            num_common_prefixes = torch.clamp(num_common_prefixes, min=1)
        num_new_accept_tokens = num_common_prefixes - (ids_accept_tokens - min(ids_accept_tokens))
        ids_accept_tokens = torch.clamp(num_new_accept_tokens + ids_accept_tokens, max=self.ncols - 1)
        print(f'num_new_accept_tokens : {num_new_accept_tokens} ids_accept_tokens : {ids_accept_tokens}')
        ids_resampled_tokens = ids_accept_tokens + 1
        # print(f'ids_accept_tokens : {ids_accept_tokens}')
        
        # resample unconverged tokens col by col
        min_resampled_id, max_resampled_id = ids_resampled_tokens.min(), ids_resampled_tokens.max()
        for b in range(min_resampled_id, min(max_resampled_id, max_unconverged_id)):
            resampled_rows = (ids_resampled_tokens == b).nonzero()[0]
            # print(f'resampled_rows : {resampled_rows}')
            l = self.input_bins_encoded_cumsum[b - 1]
            r = self.input_bins_encoded_cumsum[b]
            if len(resampled_rows) > 0:
                resampled_tokens = self.reject_sampling_col(last_logits[resampled_rows, l:r], logits[resampled_rows, l:r], b) # the score is kept, so not to update this. 
                ids_to_input[resampled_rows, b] = resampled_tokens
        print(f'after speculative ids_to_input: {ids_to_input}')
        return num_new_accept_tokens, ids_accept_tokens, ids_to_input
    

class SampleFromModel(CardEst):
    """Sample from an autoregressive model."""

    def __init__(self, model, table, num_samples_per_query, device=None):
        super(SampleFromModel, self).__init__()
        self.model = model
        self.table = table  # The table that MADE is trained on.
        self.num_samples_per_query = num_samples_per_query
        self.device = device  #device to use for pytorch

        doms = [c.DistributionSize() for c in table.columns]
        # Right shift by 1; put 0 at head.
        doms[1:] = doms[:-1]
        doms[0] = 0
        self.cumsum_shifted_doms = np.cumsum(doms)
        print('shifted cumsum', self.cumsum_shifted_doms)

    def __str__(self):
        return 'msample_{}'.format(self.num_samples_per_query)

    def SampleTuples(self, num):
        """Samples num tuples from the MADE model"""
        samples = self.model.sample(num,
                                    self.device).to(torch.int32).cpu().numpy()
        return samples

    def Query(self, columns, operators, vals):
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)
        self.OnStart()

        # [N, num cols].
        tuples = self.SampleTuples(self.num_samples_per_query)

        # all_valids:
        # [ (col1) T, F, F, T; (col2) F, F, T; (col3) T ]
        #
        # Samples:
        # [ [ 0, 2, 0 ];  [1, 1, 0] ]
        #
        # Then only the first sample satisfies the query.

        all_valids = []
        for col, op, val in zip(columns, operators, vals):
            if op is not None:
                valid = OPS[op](col.all_distinct_values, val)
            else:
                valid = [True] * col.DistributionSize()
            all_valids.extend(valid)
        all_valids = np.asarray(all_valids)

        # all() along column dimension: indicates whether each sample passes.
        s = all_valids.take(tuples + self.cumsum_shifted_doms).all(1).sum()
        sel = s * 1.0 / self.num_samples_per_query

        self.OnEnd()
        return np.ceil(sel * self.table.cardinality).astype(dtype=np.int32)


class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None):
        super(Oracle, self).__init__()
        self.table = table
        self.limit_first_n = limit_first_n

    def __str__(self):
        return 'oracle'

    def Query(self, columns, operators, vals, return_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()

        bools = None
        for c, o, v in zip(columns, operators, vals):
            if self.limit_first_n is None:
                inds = OPS[o](c.data, v)
            else:
                # For data shifts experiment.
                inds = OPS[o](c.data[:self.limit_first_n], v)

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        self.OnEnd()
        if return_masks:
            return bools
        return c

