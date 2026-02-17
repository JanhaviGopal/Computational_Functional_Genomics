from dataclasses import dataclass
import pandas as pd
from pyfaidx import Fasta


@dataclass
class SeqAugmenter:
    fasta_path: str

    def __post_init__(self):
        self.genome = Fasta(
            self.fasta_path,
            as_raw=True,
            sequence_always_upper=True
        )

    def fetch_sequence(self, chrom: str, start: int, end: int) -> str | None:
        seq = self.genome[chrom][start:end]
        if len(seq) != (end - start) or 'N' in seq:
            return None
        return str(seq)

    def augment_tsv(self, input_tsv: str, output_tsv: str) -> None:
        df = pd.read_csv(input_tsv, sep="\t")

        df["sequence"] = [
            self.fetch_sequence(row.chr, row.start, row.end)
            for row in df.itertuples(index=False)
        ]

        df.dropna(subset=["sequence"], inplace=True)
        df.to_csv(output_tsv, sep="\t", index=False)
