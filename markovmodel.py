import pandas as pd
import itertools
from collections import defaultdict


class MarkovModelBuilder:
    """
    Builds Markov models of order 0 to m from DNA sequences
    and exports conditional probability tables to Excel.

    Alphabet: {A, C, G, T}
    """

    def __init__(self, sequences, max_order):
        """
        sequences : list of DNA sequences (strings)
        max_order : maximum Markov order (m)
        """
        self.sequences = sequences
        self.max_order = max_order
        self.alphabet = ["A", "C", "G", "T"]

    def _all_kmers(self, k):
        """Generate all possible k-mers"""
        return ["".join(p) for p in itertools.product(self.alphabet, repeat=k)]

    def _count_kmers(self, k):
        """
        Count k-mers across all sequences.

        For k = m+1 → numerator
        For k = m   → denominator
        """
        counts = defaultdict(int)

        for seq in self.sequences:
            L = len(seq)
            for i in range(L - k + 1):
                kmer = seq[i:i+k]
                counts[kmer] += 1

        return counts

    def build_probability_tables(self):
        """
        Build conditional probability tables for all orders
        from 0 to max_order.

        Returns:
            dict: {order -> pandas DataFrame}
        """
        tables = {}

        for order in range(self.max_order + 1):
            # Order k Markov → (k+1)-mers
            k = order
            m = k + 1

            """
            Mathematical definition:

            P(x_t | x_{t-k}...x_{t-1})
            = count(x_{t-k}...x_t)
              --------------------------------
              sum_y count(x_{t-k}...x_{t-1}y)
            """

            # Count (k+1)-mers and k-mers
            higher_counts = self._count_kmers(m)
            lower_counts = self._count_kmers(k) if k > 0 else None

            rows = []

            if k == 0:
                # 0th-order Markov (i.i.d model)
                total = sum(higher_counts.values())
                for base in self.alphabet:
                    prob = higher_counts.get(base, 0) / total
                    rows.append({
                        "Context": "∅",
                        "Next": base,
                        "Probability": prob
                    })
            else:
                contexts = self._all_kmers(k)
                for context in contexts:
                    denom = sum(
                        higher_counts.get(context + b, 0)
                        for b in self.alphabet
                    )

                    # Skip unseen contexts
                    if denom == 0:
                        continue

                    for base in self.alphabet:
                        prob = higher_counts.get(context + base, 0) / denom
                        rows.append({
                            "Context": context,
                            "Next": base,
                            "Probability": prob
                        })

            df = pd.DataFrame(rows)
            tables[order] = df

        return tables

    def export_to_excel(self, output_file):
        """
        Export probability tables to Excel.
        One sheet per Markov order.
        """
        tables = self.build_probability_tables()

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for order, df in tables.items():
                sheet_name = f"Order_{order}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    # Example usage
    sequences = [
        "ACGTAGCTAGCTAGCTAG",
        "CGTACGTAGCTAGCTAGC",
        "GTAGCTAGCTAGCTAGCT"
    ]
    max_order = 2
    model = MarkovModelBuilder(sequences, max_order)
    model.export_to_excel("output.xlsx")
    