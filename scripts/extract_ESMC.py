import os
import torch
import argparse
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
import pandas as pd


# Usage:
#  python scripts/extract_ESMC.py -i "bench_IL-4 release.csv" -m esmc-600m -o esmc_bench_il4.pt

LENGTH_LIMIT = 2048


def assign_protein_sequence(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    def get_protein_sequence(row, uniprot_sequences=uniprot_sequences, ncbi_sequences=ncbi_sequences):
        if row['Protein ID'] in uniprot_sequences:
            return uniprot_sequences[row['Protein ID']]
        else:
            return ncbi_sequences[row['Protein ID']]

    df['Protein Seq'] = df.apply(get_protein_sequence, axis=1)
    return df


def preprocess_positions(df: pd.DataFrame) -> pd.DataFrame:
    df['Starting Position'] = df['Starting Position'] - 1  # корректируем в нулевую индексацию
    return df


def full_preprocessing_pipeline(df_path: str, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    df = assign_protein_sequence(df, uniprot_sequences, ncbi_sequences)
    df = preprocess_positions(df)
    return df


def shorten_protein_sequence(df: pd.DataFrame, length_limit: int = LENGTH_LIMIT) -> pd.DataFrame:
    updated_rows = []

    def replace_substring_with_updated_coords(sequence: str, start: int, end: int, replacement: str):
        new_sequence = sequence[:start] + replacement + sequence[end:]
        new_start = start
        new_end = start + len(replacement)
        return new_sequence, new_start, new_end

    for _, row in df.iterrows():
        sequence = row['Protein Seq']
        start = int(row['Starting Position'])
        end = int(row['Ending Position'])
        num_changes = row['Num changes']
        epitope_seq = row['Epitope Seq'].replace('-', '')

        # Проверяем, если длина последовательности больше лимита
        if len(sequence) > length_limit:
            length_epitope = end - start
            context = length_limit - length_epitope
            left = max(0, start - context // 2)
            right = min(len(sequence), end + context // 2)

            new_start = start - left
            new_end = new_start + length_epitope

            sequence = sequence[left:right]
        else:
            new_start = start
            new_end = end

        # Если есть изменения в эпитопе, вставляем новую последовательность
        if num_changes > 0:
            sequence, new_start, new_end = replace_substring_with_updated_coords(sequence, new_start, new_end, epitope_seq)

        # Обновляем строку
        updated_row = row.copy()
        assert sequence[new_start:new_end] == epitope_seq
        updated_row['Protein Seq'] = sequence
        updated_row['New Starting Position'] = new_start
        updated_row['New Ending Position'] = new_end

        updated_rows.append(updated_row)

    updated_df = pd.DataFrame(updated_rows)
    updated_df['ID'] = updated_df[['Epitope ID', 'Starting Position', 'Ending Position', 'New Starting Position', 'New Ending Position']].astype(str).apply('_'.join, axis=1)
    return updated_df


class DataLoader:
    """
    Data loader for reading a FASTA file and creating batches based on a token limit.

    Args:
    - fasta_file (str): Path to the FASTA file.
    - batch_token_limit (int, optional): Maximum number of tokens per batch. Defaults to 4096.
    - model (object): Model object with a `_tokenize` method for tokenizing sequences.
    """
    def __init__(self, df, model, batch_token_limit=LENGTH_LIMIT + 2):
        self.df = df
        self.batch_token_limit = batch_token_limit
        self.model = model
        self.sequences = df['Protein Seq']
        self.id = df['ID']
        self.total_sequences = len(self.df)

    def __len__(self):
        # Approximate total number of batches
        total_tokens = sum(len(seq) + 2 for seq in self.sequences)  # +2 for BOS and EOS tokens
        return (total_tokens + self.batch_token_limit - 1) // self.batch_token_limit

    def __iter__(self):
        ids, lengths, seqs = [], [], []
        current_token_count = 0

        for i, seq in enumerate(self.sequences):
            seq_length = len(seq)
            token_count = seq_length + 2  # Include BOS and EOS tokens
            if current_token_count + token_count > self.batch_token_limit and ids:
                # Yield current batch if adding the new sequence exceeds the token limit
                tokens = self.model._tokenize(seqs)
                yield ids, lengths, tokens
                ids, lengths, seqs = [], [], []
                current_token_count = 0

            # Add the current sequence to the batch
            ids.append(self.id.iloc[i])
            lengths.append(seq_length)
            seqs.append(seq)
            current_token_count += token_count

        # Yield any remaining sequences
        if ids:
            tokens = self.model._tokenize(seqs)
            yield ids, lengths, tokens


def extract_mean_representations(model, df):
    mean_representations = {}
    data_loader = DataLoader(df, model=model)

    with torch.no_grad():  # Disable gradient calculations
        for batch_ids, batch_lengths, batch_tokens in tqdm(data_loader, desc="Processing batches", leave=False):
            output = model(batch_tokens)
            logits, embeddings, hiddens = (
                output.sequence_logits,
                output.embeddings,
                output.hidden_states,
            )

            for i, ID in enumerate(batch_ids):
            # Extract the last hidden states for the sequence
                representations = embeddings[i, 1:batch_lengths[i]+1, :].detach().to('cpu')

                ids = ID.split('_')
                new_start = int(ids[-2])
                new_end = int(ids[-1])

                representations = representations[new_start:new_end, :]

                ID = '_'.join(ID.split('_')[:-2])

                # compute mean representation of the sequence
                mean_representations[ID] = representations.mean(dim=0)

    return mean_representations


def parse_fasta_to_dict(file_path: str) -> dict[str, str]:
    fasta_dict = {}
    # Parse the FASTA file and print details
    for record in SeqIO.parse(file_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a FASTA file")
    parser.add_argument("-i", "--data_name", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("-m", "--model_checkpoint", type=str, required=True, help="Model checkpoint identifier")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    # Define the input parameters
    data_name = args.data_name
    model_checkpoint = args.model_checkpoint
    output_file = args.output

    # Define the device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load the model based on the checkpoint identifier
    if model_checkpoint == 'esmc-300m':
        model = ESMC.from_pretrained("esmc_300m").to(device)  # "cuda" or "cpu"
        print("Model transferred to device:", model.device)

    elif model_checkpoint == 'esmc-600m':
        model = ESMC.from_pretrained("esmc_600m").to(device)
        print("Model transferred to device:", model.device)
    else:
        print("Model not found!")
        print("Choose a valid model checkpoint: 'esmc-300m' or 'esmc-600m'")
        exit(1)

    project_dir = '/mnt/nfs_protein/gavrilenko/vaccine-design/'
    uniprot_sequences = parse_fasta_to_dict(project_dir + 'sequences.fasta')
    ncbi_sequences = parse_fasta_to_dict(project_dir + 'ncbi_sequences.fasta')

    csv_dir = '/mnt/nfs_protein/gavrilenko/vaccine-design/cytokine/parent_epitopes'
    df_path = os.path.join(csv_dir, data_name)

    embed_dir = '/mnt/nfs_protein/gavrilenko/vaccine-design/cytokine/embeddings/'
    embed_path = os.path.join(embed_dir, output_file)

    df = full_preprocessing_pipeline(df_path, uniprot_sequences, ncbi_sequences)
    chunked_dataset = shorten_protein_sequence(df)

    # Extract representations
    result = extract_mean_representations(model, chunked_dataset)

    assert len(df) == len(result)
    # Save results
    torch.save(result, embed_path)

    print(f'Process Finished! Results saved to {embed_path}')


if __name__ == "__main__":
    main()
