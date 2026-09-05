"""Document encoders for the reference implementation (torch)."""

MODELS = {
    "mini": "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    "base": "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    "multilingual": "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
}


def resolve(model):
    return MODELS.get(model, model)


class TorchEncoder:
    def __init__(self, model, max_seq=256, device=None):
        import torch
        from sentence_transformers import SparseEncoder
        self.model_id = resolve(model)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"trust_remote_code": True} if "gte" in self.model_id or "multilingual" in self.model_id else {}
        self.m = SparseEncoder(self.model_id, device=device, **kw)
        self.m.max_seq_length = max_seq

    def qlut(self):
        return self.m[0].sub_modules["query"][0].weight.detach().float().cpu().numpy().ravel()

    def vocab(self):
        return self.m.tokenizer.convert_ids_to_tokens(list(range(len(self.qlut()))))

    def encode(self, texts, batch_size=32, wmin=0.01):
        """returns list[dict[term_id, weight]]"""
        emb = self.m.encode_document(texts, batch_size=batch_size,
                                     convert_to_sparse_tensor=True, show_progress_bar=False).coalesce()
        idx, val = emb.indices().cpu().numpy(), emb.values().cpu().numpy()
        out = [dict() for _ in texts]
        keep = val >= wmin
        for r, c, v in zip(idx[0][keep], idx[1][keep], val[keep]):
            out[r][int(c)] = float(v)
        return out
