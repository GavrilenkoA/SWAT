#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_ESMC_parallel.py
========================
Извлечение средних эмбеддингов эпитопов модели ESM-C (300 M / 600 M)
с Data-Parallel инференсом на нескольких GPU.  Тип данных полностью
сохраняем float32.

Запуск:
    python extract_ESMC_parallel.py \
        -i "bench_IL-4 release.csv" \
        -m esmc-600m \
        -o esmc_bench_il4.pt \
        --devices 0,1
"""

import os
import argparse
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO

from esm.models.esmc import ESMC  # pip install fair-esm>=2.0


# --------------------------------------------------------------------------- #
#                                ПАРАМЕТРЫ                                    #
# --------------------------------------------------------------------------- #

LENGTH_LIMIT = 2048                # максимум a.a. после обрезки
BATCH_TOKEN_LIMIT = LENGTH_LIMIT + 2  # лимит токенов на батч (+ BOS/EOS)


# --------------------------------------------------------------------------- #
#                          ПРЕПРОЦЕССИНГ ДАННЫХ                                #
# --------------------------------------------------------------------------- #

def assign_protein_sequence(
    df: pd.DataFrame, uniprot: Dict[str, str], ncbi: Dict[str, str]
) -> pd.DataFrame:
    def _get(row):
        pid = row["Protein ID"]
        return uniprot.get(pid, ncbi.get(pid, ""))
    df["Protein Seq"] = df.apply(_get, axis=1)
    return df


def preprocess_positions(df: pd.DataFrame) -> pd.DataFrame:
    df["Starting Position"] = df["Starting Position"] - 1
    return df


def full_preprocessing_pipeline(
    csv_path: str, uniprot: Dict[str, str], ncbi: Dict[str, str]
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = assign_protein_sequence(df, uniprot, ncbi)
    df = preprocess_positions(df)
    return df


def shorten_protein_sequence(
    df: pd.DataFrame, limit: int = LENGTH_LIMIT
) -> pd.DataFrame:
    """Обрезает белок до LIMIT так, чтобы эпитоп остался внутри."""
    upd_rows: List[pd.Series] = []

    def _replace(seq: str, st: int, en: int, repl: str) -> Tuple[str, int, int]:
        new_seq = seq[:st] + repl + seq[en:]
        return new_seq, st, st + len(repl)

    for _, row in df.iterrows():
        seq = row["Protein Seq"]
        st, en = int(row["Starting Position"]), int(row["Ending Position"])
        num_changes = row["Num changes"]
        ep_seq = row["Epitope Seq"].replace("-", "")

        if len(seq) > limit:
            epi_len = en - st
            ctx = limit - epi_len
            left = max(0, st - ctx // 2)
            right = min(len(seq), en + ctx // 2)
            new_st = st - left
            new_en = new_st + epi_len
            seq = seq[left:right]
        else:
            new_st, new_en = st, en

        if num_changes > 0:
            seq, new_st, new_en = _replace(seq, new_st, new_en, ep_seq)

        assert seq[new_st:new_en] == ep_seq, "Epitope mismatch after trimming"

        r = row.copy()
        r["Protein Seq"] = seq
        r["New Starting Position"] = new_st
        r["New Ending Position"] = new_en
        upd_rows.append(r)

    out = pd.DataFrame(upd_rows)
    out["ID"] = out[
        ["Epitope ID", "Starting Position", "Ending Position",
         "New Starting Position", "New Ending Position"]
    ].astype(str).agg("_".join, axis=1)
    return out


# --------------------------------------------------------------------------- #
#                             DATA LOADER
# --------------------------------------------------------------------------- #

class DataLoader:
    """
    Итерируется по DataFrame, выдавая батчи
    (ids, lengths, token_tensor), где число токенов ≤ batch_token_limit.
    """

    def __init__(
        self, df: pd.DataFrame, model: torch.nn.Module,
        batch_token_limit: int = BATCH_TOKEN_LIMIT
    ):
        # достаём «нутро» модели, где есть _tokenize
        base = model.module if isinstance(model, torch.nn.DataParallel) else model
        if hasattr(base, "model"):   # EmbeddingWrapper
            base = base.model
        self.tokenizer_model = base

        self.ids = df["ID"].tolist()
        self.seqs = df["Protein Seq"].tolist()
        self.batch_token_limit = batch_token_limit

    def __iter__(self):
        ids, lens, seqs, tok_cnt = [], [], [], 0

        for sid, seq in zip(self.ids, self.seqs):
            n_tok = len(seq) + 2
            if tok_cnt + n_tok > self.batch_token_limit and ids:
                tokens = self.tokenizer_model._tokenize(seqs)  # type: ignore
                yield ids, lens, tokens
                ids, lens, seqs, tok_cnt = [], [], [], 0

            ids.append(sid)
            lens.append(len(seq))
            seqs.append(seq)
            tok_cnt += n_tok

        if ids:
            tokens = self.tokenizer_model._tokenize(seqs)  # type: ignore
            yield ids, lens, tokens


# --------------------------------------------------------------------------- #
#                          ИНФЕРЕНС НА НЕСК. GPU                               #
# --------------------------------------------------------------------------- #

def extract_mean_representations(
    model: torch.nn.Module, df: pd.DataFrame
) -> Dict[str, Tensor]:
    """Считает средний вектор эпитопа из embeddings."""
    mean_repr: Dict[str, Tensor] = {}
    loader = DataLoader(df, model)

    # где находится «главная» копия весов
    main_dev = (
        model.device_ids[0] if isinstance(model, torch.nn.DataParallel)
        else next(model.parameters()).device
    )

    with torch.no_grad():
        for batch_ids, batch_lens, batch_tok in tqdm(loader, desc="Batches"):
            batch_tok = batch_tok.to(main_dev)
            embeddings, = model(batch_tok)      # -> Tensor (B, L+2, C)

            for i, full_id in enumerate(batch_ids):
                rep = embeddings[i, 1:batch_lens[i]+1, :].cpu()

                *base_tokens, new_st, new_en = full_id.split("_")
                new_st, new_en = int(new_st), int(new_en)
                epi_rep = rep[new_st:new_en, :]

                clean_id = "_".join(base_tokens)
                mean_repr[clean_id] = epi_rep.mean(dim=0)

    return mean_repr


# --------------------------------------------------------------------------- #
#                              I/O УТИЛИТЫ
# --------------------------------------------------------------------------- #

def parse_fasta_to_dict(path: str) -> Dict[str, str]:
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(path, "fasta")}


# --------------------------------------------------------------------------- #
#                       ЗАГРУЗКА И ОБЁРТКА МОДЕЛИ
# --------------------------------------------------------------------------- #

class EmbeddingWrapper(torch.nn.Module):
    """
    Обёртка, которая берёт ESM-C, а в forward
    возвращает только embeddings как кортеж (Tensor,).
    Такой кортеж DataParallel сумеет собрать.
    """
    def __init__(self, model: ESMC):
        super().__init__()
        self.model = model

    def forward(self, tokens: Tensor):
        out = self.model(tokens)
        return (out.embeddings,)          # именно кортеж!


def load_model(checkpoint: str, device_ids: List[int]) -> torch.nn.Module:
    ckpt_map = {"esmc-300m": "esmc_300m", "esmc-600m": "esmc_600m"}
    if checkpoint not in ckpt_map:
        raise ValueError("checkpoint must be 'esmc-300m' or 'esmc-600m'")
    base = ESMC.from_pretrained(ckpt_map[checkpoint]).eval()  # float32
    base = base.to(f"cuda:{device_ids[0]}")                   # главная GPU

    wrapped = EmbeddingWrapper(base)
    if len(device_ids) > 1:
        wrapped = torch.nn.DataParallel(wrapped, device_ids=device_ids)
        print(f"Model wrapped with DataParallel on GPUs {device_ids}")
    else:
        print(f"Model on cuda:{device_ids[0]}")
    return wrapped


# --------------------------------------------------------------------------- #
#                                   MAIN
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Extract mean ESM-C embeddings with multi-GPU data parallel"
    )
    parser.add_argument("-i", "--data_name", required=True, help="CSV-файл")
    parser.add_argument("-m", "--model_checkpoint", required=True,
                        choices=["esmc-300m", "esmc-600m"])
    parser.add_argument("-o", "--output", required=True, help=".pt-файл вывода")
    parser.add_argument("--devices", default="0,1",
                        help="Список GPU через запятую (по умолчанию 0,1)")
    args = parser.parse_args()

    # ---------- GPU SETUP ----------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA не доступна")
    dev_ids = [int(d) for d in args.devices.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, dev_ids))
    torch.cuda.set_device(dev_ids[0])

    model = load_model(args.model_checkpoint, dev_ids)

    # ---------- ПУТИ К ДАННЫМ ----------
    project_dir = "/mnt/nfs_protein/gavrilenko/vaccine-design/"
    uniprot = parse_fasta_to_dict(project_dir + "sequences.fasta")
    ncbi = parse_fasta_to_dict(project_dir + "ncbi_sequences.fasta")

    csv_dir = project_dir + "cytokine/parent_epitopes/"
    df_path = os.path.join(csv_dir, args.data_name)

    embed_dir = project_dir + "cytokine/embeddings/"
    os.makedirs(embed_dir, exist_ok=True)
    embed_path = os.path.join(embed_dir, args.output)

    # ---------- PIPELINE ----------
    df_raw = full_preprocessing_pipeline(df_path, uniprot, ncbi)
    df_trim = shorten_protein_sequence(df_raw)

    reps = extract_mean_representations(model, df_trim)
    assert len(df_raw) == len(reps), "Не все эпитопы обработаны"

    torch.save(reps, embed_path)
    print(f"Готово! Эмбеддинги сохранены в {embed_path}")


if __name__ == "__main__":
    main()
