import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class TemporalCachedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_kv_past = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x, past_cond, cond, k_cache=None, v_cache=None, causal_mask=None):
        q = self.to_q(x)
        
        k, v = self.to_kv(cond).chunk(2, dim=-1)

        if past_cond is not None:
            past_k, past_v = self.to_kv_past(past_cond).chunk(2, dim=-1)
            k, v = torch.concat([past_k, k], dim=1), torch.concat([past_v, v], dim=1)
        
        if k_cache is not None and v_cache is not None:
            k, v = torch.concat([k_cache, k], dim=1), torch.concat([v_cache, v], dim=1)
            
        out_k = k.clone()
        out_v = v.clone()

        q = rearrange(q, "B T (h d) -> B h T d", h=self.heads)
        k = rearrange(k, "B T (h d) -> B h T d", h=self.heads)
        v = rearrange(v, "B T (h d) -> B h T d", h=self.heads)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        if causal_mask is None:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)
        else:     
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False, attn_mask=causal_mask)

        x = rearrange(x, "B h T d -> B T (h d)")
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        
        return x, out_k, out_v