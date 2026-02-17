# from seqaugmenter import SeqAugmenter

# augmenter = SeqAugmenter("genome/hg38.fa")
# path = f"G:\PhD_2026\Assignments\CFG_Assignments\markov_tf_binding\genome\\"
# augmenter.augment_tsv(
#     input_tsv=f"{path}chr1_200bp_bins.tsv",
#     output_tsv=f"{path}chr1_200bp_bins_with_seq.tsv"
# )

from fasta import GenomeFetcher
from markov import MarkovModel, cross_validate_markov, select_best_m, plot_roc, plot_score_histogram
import pandas as pd

#
# Load augmented TSV
df = pd.read_csv("./genome/chr1_200bp_bins_with_seq.tsv", sep="\t")

# Create label
df["label"] = (df["CTCF"] == "B").astype(int)
df = df[["sequence", "label"]]

# Choose best Markov order
m_star, aucs = select_best_m(df, m_values=range(2, 9), k=5)
print("Optimal m:", m_star)

# Final evaluation with best m
labels, scores = cross_validate_markov(df, m_star, k=5)

plot_roc(labels, scores, title=f"ROC (m={m_star})")
plot_score_histogram(labels, scores, title=f"Score Distribution (m={m_star})")

