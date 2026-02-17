from pyfaidx import Fasta

class GenomeFetcher:
    def __init__(self, fasta_path):
        self.genome = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)

    def fetch(self, chrom, start, end):
        seq = self.genome[chrom][start:end]
        if 'N' in seq:
            return None
        return str(seq)

    def close(self):
        self.genome.close()

genome = GenomeFetcher("genome/hg38.fa")
seq = genome.fetch("chr3", 12200, 12400)
print(len(seq))
genome.close()
print(seq)