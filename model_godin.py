import torch
import torch.nn as nn


class GODINHead(nn.Module):
    """
    G-ODIN Head:
    h(x): cosine similarity with learnable class weights
    g(x): linear + LayerNorm + Sigmoid

    Output:
      logits = h / g  (for CE loss)
      also returns (h, g) for optional OOD scoring
    """

    def __init__(self, feat_dim: int, num_classes: int, clamp_eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.clamp_eps = clamp_eps

        # Cosine similarity class weights (K, D)
        self.class_weight = nn.Parameter(torch.randn(num_classes, feat_dim))

        # Domain branch
        self.g_linear = nn.Linear(feat_dim, 1)
        self.g_bn = nn.LayerNorm(1)

        # Initialization
        nn.init.normal_(self.class_weight, std=0.02)
        nn.init.xavier_uniform_(self.g_linear.weight)
        nn.init.zeros_(self.g_linear.bias)

    def forward(self, feat: torch.Tensor):
        """
        feat: (B, D) features from the backbone
        return: logits (B, K), h (B, K), g (B, 1)
        """
        # --- h(x): cosine similarity ---
        f = feat
        w = self.class_weight
        f_norm = f / (f.norm(dim=1, keepdim=True) + 1e-8)
        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-8)
        h = torch.matmul(f_norm, w_norm.t())

        # --- g(x): linear + LN + Sigmoid ---
        g = self.g_linear(f)
        g = self.g_bn(g)
        g = torch.sigmoid(g).clamp(min=self.clamp_eps,
                                   max=1.0 - self.clamp_eps)

        # --- logits = h / g ---
        logits = h / g
        return logits, h, g


class GODINSequential(nn.Module):
    """
    Wrapper: TITAN backbone + G-ODIN head
    If return_scores=True, returns (logits, h, g)
    Otherwise returns logits only.
    """

    def __init__(self, model, head: nn.Module, return_scores: bool = False):
        super(GODINSequential, self).__init__()
        self.model = model
        self.head = head
        self.return_scores = return_scores

    def forward(self, *args, **kwargs):
        # TITAN returns (B, 768)
        feat = self.model.encode_slide_from_patch_features(*args, **kwargs)
        logits, h, g = self.head(feat)

        if self.return_scores:
            return logits, h, g
        return logits
