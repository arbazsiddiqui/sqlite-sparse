"""Write the .sprs sidecar: the MLM head llama.cpp's converter drops, the static
query weight table and the vocabulary. Little-endian throughout.

  'SPRS' | u32 version=1 | u32 hidden | u32 vocab_n | u32 weight fmt (0=f32)
  | u32 activation (0 log1p(relu), 1 log1p(log1p(relu)) for v3) | 8 pad
  | dense.W (H*H f32) | dense.b (H) | ln.gamma (H) | ln.beta (H)
  | decoder.W (vocab_n*H f32, row-major) | decoder.b (vocab_n)
  | qlut (vocab_n f32) | u32 len | vocab blob (tokens joined by \n)
"""
import struct
import sys

import numpy as np


def extract(model_id, out_path, act_flag=0):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from sentence_transformers import SparseEncoder

    se = SparseEncoder(model_id, device="cpu")
    qlut = se[0].sub_modules["query"][0].weight.detach().float().numpy().ravel()
    m = AutoModelForMaskedLM.from_pretrained(model_id)
    sd = {k: v.detach().float().numpy() for k, v in m.state_dict().items()}
    H = sd["cls.predictions.transform.dense.weight"].shape[0]
    dec_w = sd["bert.embeddings.word_embeddings.weight"]
    vocab_n = dec_w.shape[0]
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = tok.convert_ids_to_tokens(list(range(vocab_n)))
    vocab_blob = "\n".join(vocab).encode("utf-8")

    with open(out_path, "wb") as f:
        f.write(b"SPRS")
        f.write(struct.pack("<IIIII8x", 1, H, vocab_n, 0, act_flag))
        for arr in (sd["cls.predictions.transform.dense.weight"],
                    sd["cls.predictions.transform.dense.bias"],
                    sd["cls.predictions.transform.LayerNorm.weight"],
                    sd["cls.predictions.transform.LayerNorm.bias"],
                    dec_w,
                    sd["cls.predictions.bias"],
                    qlut.astype(np.float32)):
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        f.write(struct.pack("<I", len(vocab_blob)))
        f.write(vocab_blob)
    import os
    print(f"wrote {out_path}: H={H} vocab={vocab_n} bytes={os.path.getsize(out_path)}")
    return H, vocab_n


if __name__ == "__main__":
    extract(sys.argv[1] if len(sys.argv) > 1
            else "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
            sys.argv[2] if len(sys.argv) > 2 else "mini.sprs")
