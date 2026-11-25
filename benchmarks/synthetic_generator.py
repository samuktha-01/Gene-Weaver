"""Deterministic synthetic genomic dataset generator for reproducible GeneWeaver benchmarks.

Produces controlled pseudo-random DNA sequences with calibrated GC content and embedded
on-target and off-target candidate sites (forward and reverse strands, varying mismatch
counts, canonical and alternate PAM motifs, and chunk boundary positions) using fixed random
seeds to ensure reproducible performance profiling without requiring real reference genomes.
"""

from pathlib import Path
import random
from typing import List, Optional, Tuple, Union

import numpy as np

from geneweaver.alignment.reverse_complement import reverse_complement


class SyntheticGenomicGenerator:
    """Generates deterministic synthetic FASTA reference datasets for benchmarking."""

    BASES = np.array(["A", "C", "G", "T"])
    # Default human genomic nucleotide background probabilities (~41% GC)
    BASE_PROBS = np.array([0.295, 0.205, 0.205, 0.295])

    @classmethod
    def generate_sequence(
        cls,
        length_bp: int,
        seed: int = 42,
    ) -> str:
        """Generates a pseudo-random DNA sequence string of specified length.

        Args:
            length_bp: Desired sequence length in base pairs.
            seed: Integer random seed for deterministic reproduction.

        Returns:
            Uppercase canonical DNA nucleotide string (A, C, G, T).
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(4, size=length_bp, p=cls.BASE_PROBS)
        chars = cls.BASES[indices]
        return "".join(chars)

    @classmethod
    def create_benchmark_fasta(
        cls,
        output_path: Union[Path, str],
        length_bp: int = 100_000,
        guide_spacer: str = "GCTTATCCAGTCTGCTACGC",
        seed: int = 42,
        header_identifier: str = "synthetic_benchmark_chr",
    ) -> Path:
        """Creates a synthetic FASTA file with embedded CRISPR targets at known positions.

        Embeds:
        - 1 Canonical on-target site on forward strand (0 MM, PAM: 'TGG')
        - 1 Canonical on-target site on reverse strand (0 MM, PAM: 'AGG' on reverse)
        - 1 Candidate with 1 mismatch at 5' distal position (PAM: 'CGG', forward)
        - 1 Candidate with 2 mismatches in non-seed region (PAM: 'TGG', reverse)
        - 1 Candidate with 3 mismatches (2 distal, 1 seed; PAM: 'GGG', forward)
        - 1 Candidate with 4 mismatches (PAM: 'AGG', reverse)
        - Sites specifically positioned to cross chunk boundaries (e.g. near 100k, 250k).

        Args:
            output_path: Destination path for the generated FASTA file.
            length_bp: Total length of the synthetic chromosome in base pairs.
            guide_spacer: 20 nt guide spacer sequence.
            seed: Random seed for deterministic generation.
            header_identifier: FASTA record identifier.

        Returns:
            Resolved Path of the written FASTA file.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if len(guide_spacer) != 20:
            raise ValueError(f"guide_spacer must be exactly 20 nt, got {len(guide_spacer)}")

        raw_seq = list(cls.generate_sequence(length_bp, seed=seed))

        # Helper to embed a 23 nt target at a specific index
        def _embed(idx: int, target_23: str) -> None:
            if 0 <= idx <= len(raw_seq) - len(target_23):
                for offset, char in enumerate(target_23):
                    raw_seq[idx + offset] = char

        # 1. On-target forward site (0 MM + TGG PAM)
        # Position around 5% of sequence
        pos_on_fwd = min(5_000, max(100, length_bp // 20))
        _embed(pos_on_fwd, guide_spacer + "TGG")

        # 2. On-target reverse site (0 MM + AGG PAM in reverse orientation)
        # On reverse strand, genomic sequence is reverse_complement(guide_spacer + PAM)
        # Reverse complement of guide_spacer + AGG
        rev_23_0mm = reverse_complement(guide_spacer + "AGG")
        pos_on_rev = min(15_000, max(200, length_bp // 10))
        _embed(pos_on_rev, rev_23_0mm)

        # 3. 1 Mismatch candidate on forward strand (Distal mismatch at pos 1: G -> A)
        pos_1mm = min(25_000, max(300, length_bp // 5))
        m1_spacer = ("A" if guide_spacer[0] != "A" else "C") + guide_spacer[1:]
        _embed(pos_1mm, m1_spacer + "CGG")

        # 4. 2 Mismatches candidate on reverse strand (Mismatches at pos 1, 2)
        pos_2mm = min(50_000, max(400, length_bp // 3))
        m2_spacer = (
            ("A" if guide_spacer[0] != "A" else "C")
            + ("T" if guide_spacer[1] != "T" else "G")
            + guide_spacer[2:]
        )
        _embed(pos_2mm, reverse_complement(m2_spacer + "TGG"))

        # 5. 3 Mismatches candidate on forward strand (Pos 1, 2, 15)
        pos_3mm = min(75_000, max(500, length_bp // 2))
        m3_list = list(guide_spacer)
        m3_list[0] = "A" if guide_spacer[0] != "A" else "C"
        m3_list[1] = "T" if guide_spacer[1] != "T" else "G"
        m3_list[14] = "A" if guide_spacer[14] != "A" else "T"
        _embed(pos_3mm, "".join(m3_list) + "GGG")

        # 6. 4 Mismatches candidate on reverse strand (Pos 1, 2, 3, 18)
        pos_4mm = min(90_000, max(600, (length_bp * 3) // 4))
        m4_list = list(guide_spacer)
        m4_list[0] = "A" if guide_spacer[0] != "A" else "C"
        m4_list[1] = "T" if guide_spacer[1] != "T" else "G"
        m4_list[2] = "C" if guide_spacer[2] != "C" else "G"
        m4_list[17] = "T" if guide_spacer[17] != "T" else "A"
        _embed(pos_4mm, reverse_complement("".join(m4_list) + "AGG"))

        # 7. Boundary test candidate crossing potential 100k chunk boundary if length >= 120k
        if length_bp >= 120_000:
            pos_boundary = 99_990  # Straddles 100,000 bp chunk border
            _embed(pos_boundary, guide_spacer + "CGG")

        final_seq = "".join(raw_seq)

        # Write FASTA formatted with 80 characters per line
        with open(out_p, mode="wt", encoding="utf-8") as handle:
            handle.write(f">{header_identifier} length={length_bp} seed={seed}\n")
            line_width = 80
            for i in range(0, len(final_seq), line_width):
                handle.write(final_seq[i : i + line_width] + "\n")

        return out_p
