import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings
from transformers import AutoTokenizer

def compute_dynamic_attn_weights(
    attentions: torch.Tensor,
    attn_slices: dict[str, slice],
    pooling: str = 'mean'
) -> dict[str, float]:
    """
    Given last‐layer attentions (batch, heads, seq, seq) and a dict of named slices,
    compute each slice’s average attention mass and normalize to sum=1.
    """
    # average over heads → (batch, seq, seq)
    attn = attentions.mean(dim=1)

    # pool each slice into a scalar
    raw: dict[str, float] = {}
    for name, sl in attn_slices.items():
        seg = attn[:, sl, :]  # (batch, seg_len, seq)
        if pooling == 'mean':
            # mean over (seg_len x seq) then batch
            raw[name] = seg.mean(dim=(1,2)).mean().item()
        else:
            raw[name] = seg.sum(dim=(1,2)).mean().item()

    total = sum(raw.values()) or 1.0
    return {name: v/total for name, v in raw.items()}


def attention_loss(
    attentions: torch.Tensor,
    attn_slices: dict[str, slice],
    pooling: str,
    weights: dict[str, float]
) -> torch.Tensor:
    """
    Compute weighted sum of pooled attentions over each named slice.
    Returns a Tensor of shape (batch,) giving the penalty for each example.
    """
    # collapse heads → (batch, seq, seq)
    attn = attentions.mean(dim=1)
    batch_size = attn.size(0)
    out = torch.zeros(batch_size, device=attn.device)

    for name, sl in attn_slices.items():
        seg = attn[:, sl, :]  # (batch, seg_len, seq)
        if pooling == 'mean':
            v = seg.mean(dim=(1,2))
        else:
            v = seg.sum(dim=(1,2))
        out = out + weights.get(name, 0.0) * v

    return out  # (batch,)


def token_gradients(
    model,
    input_ids: torch.LongTensor,
    attn_slices: dict[str, slice],
    target_slice: slice,
    target_weight: float = 1.0,
    attn_pool: str = 'mean',
    attn_weight: float = 0.5
):
    """
    Compute gradients w.r.t. control tokens by blending:
      - CE loss on `target_slice`
      - an attention penalty over the segments in `attn_slices`
    """
    # 1) ensure batch dim
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # 2) pull out our slices
    control_slice = attn_slices['control']

    # 3) get embeddings & build one-hot in matching dtype/device
    embed_weights = get_embedding_matrix(model)          # (vocab, dim)
    dtype, device = embed_weights.dtype, embed_weights.device

    # 4) get “clean” embeddings for everything once
    embeds = get_embeddings(model, input_ids).detach()   # (1, seq, dim)

    # 5) build one-hot only over control tokens WITH in-place ops
    ctrl_ids    = input_ids[0, attn_slices['control']]         # (control_len,)
    control_len = ctrl_ids.size(0)
    # start with a float tensor (leaf, no grad yet)
    one_hot = torch.zeros(
        (control_len, embed_weights.size(0)),
        dtype=dtype,
        device=device
    )
    # in-place scatter so this remains a leaf
    one_hot.scatter_(1, ctrl_ids.unsqueeze(1), 1.0)
    # now mark it to require gradients
    one_hot.requires_grad_()

    # 6) splice & forward as before…
    control_embeds = (one_hot @ embed_weights).unsqueeze(0)
    full_embeds    = torch.cat([
        embeds[:, :control_slice.start, :],
        control_embeds,
        embeds[:, control_slice.stop:, :]
    ], dim=1)

    # 7) forward with attentions
    out = model(inputs_embeds=full_embeds, output_attentions=True)
    logits = out.logits                                  # (1, seq, vocab)
    attentions = out.attentions[-1]                      # (1, heads, seq, seq)

    # 8) CE loss on the true target suffix
    tgt_ids = input_ids[0, target_slice]                 # (target_len,)
    loss_ce = nn.CrossEntropyLoss()(logits[0, target_slice, :], tgt_ids)

    # 9) dynamic attention weights + penalty
    dyn_w = compute_dynamic_attn_weights(attentions, attn_slices, pooling=attn_pool)
    loss_attn = attention_loss(attentions, attn_slices, pooling=attn_pool, weights=dyn_w)

    # 10) combine & backprop
    loss = target_weight * loss_ce + attn_weight * loss_attn.mean()
    loss.backward()

    # 11) return just the control-region gradient
    return one_hot.grad.clone()  # (control_len, vocab)



class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        input_ids = self.input_ids.to(model.device)

        attn_slices = {
            'goal':    self._goal_slice,
            'control': self._control_slice,
            # optionally: 'prefix': self._user_role_slice
        }

        return token_gradients(
            model=model,
            input_ids=input_ids,
            attn_slices=attn_slices,
            target_slice=self._target_slice,
            target_weight=1.0,
            attn_pool='mean',
            attn_weight=0.5
        )

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.inf
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=32, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False,
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        # del control_cands, loss ; gc.collect()
        gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers), control_cands, loss
