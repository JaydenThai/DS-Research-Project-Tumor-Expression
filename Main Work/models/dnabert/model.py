from itertools import product
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def build_kmer_vocab(k: int):
    bases = ["A", "C", "G", "T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    vocab = SPECIALS + kmers
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos


def seq_to_kmers(seq: str, k: int) -> List[str]:
    seq = seq.upper()
    toks: List[str] = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in "ACGT" for c in kmer):
            toks.append("[UNK]")
        else:
            toks.append(kmer)
    return ["[CLS]"] + toks + ["[SEP]"]


def encode_batch(seqs: List[str], k: int, stoi: dict, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenised = [seq_to_kmers(s, k) for s in seqs]
    if max_len is None:
        max_len = max(len(t) for t in tokenised)
    pad_id = stoi["[PAD]"]
    unk_id = stoi["[UNK]"]
    input_ids = []
    attn = []
    for toks in tokenised:
        ids = [stoi.get(t, unk_id) for t in toks[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids += [pad_id] * pad_n
            mask += [0] * pad_n
        input_ids.append(ids)
        attn.append(mask)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)


def onehot5_to_strings(x: torch.Tensor) -> List[str]:
    """Convert (B,5,L) to strings using channel order A,T,G,C,N to match PromoterDataset."""
    assert x.ndim == 3 and x.size(1) == 5, "expected (B,5,L)"
    idx = x.argmax(dim=1)
    lut = {0: "A", 1: "T", 2: "G", 3: "C", 4: "N"}
    return ["".join(lut[int(i)] for i in row) for row in idx]


class DNABertClassifier(nn.Module):
    def __init__(
        self,
        k: int = 6,
        num_labels: int = 4,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_size: Optional[int] = None,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        vocab: Optional[List[str]] = None,
        stoi: Optional[dict] = None,
    ):
        super().__init__()
        self.k = k
        self.num_labels = num_labels

        if vocab is None or stoi is None:
            vocab, stoi, _ = build_kmer_vocab(k)
        self.stoi = stoi
        self.vocab_size = len(vocab)

        if ffn_size is None:
            ffn_size = 4 * hidden_size

        self.token_embeddings = nn.Embedding(self.vocab_size, hidden_size)
        self.pos_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.emb_layer_norm = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = input_ids.ne(self.stoi["[PAD]"])

        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        x = self.emb_layer_norm(self.emb_dropout(x))

        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())

        cls = x[:, 0]
        logits = self.classifier(cls)
        return logits


def prepare_inputs_from_onehot(onehot_batch: torch.Tensor, k: int, stoi: dict, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = onehot5_to_strings(onehot_batch)
    input_ids, attn = encode_batch(seqs, k, stoi, max_len=max_len)
    return input_ids.to(onehot_batch.device), attn.to(onehot_batch.device)


