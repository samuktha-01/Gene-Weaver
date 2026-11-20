# GeneWeaver – GPU-Accelerated CRISPR Alignment Engine

**GeneWeaver** is a bioinformatics and high-performance computing project designed to identify potential CRISPR-Cas9 off-target sites in genomic sequences. It aims to combine efficient DNA sequence processing, parallel alignment, GPU acceleration, and distributed computing to make large-scale genomic searches faster.

The project starts with a CPU-based alignment engine and progressively develops toward a GPU-accelerated and distributed genomic alignment system.

---

## Project Overview

CRISPR-Cas9 is a genome-editing technology that uses a guide RNA to direct the Cas9 enzyme toward a specific DNA sequence. However, the guide RNA may also bind to similar unintended DNA locations, known as **potential off-target sites**.

GeneWeaver aims to identify these candidate locations by comparing a guide sequence against genomic DNA, detecting mismatches, and evaluating potential target sites.

The initial target is the human reference genome (GRCh38).

### Key Objectives

- Process large genomic FASTA files efficiently.
- Implement memory-efficient genomic sequence chunking.
- Encode DNA sequences using two-bit representation.
- Identify potential off-target sequences.
- Support mismatch detection and sequence alignment.
- Develop GPU-accelerated alignment using CUDA-compatible technologies.
- Enable distributed genomic processing.
- Provide an interactive terminal dashboard for monitoring execution.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| NumPy | Vectorized sequence processing |
| BioPython | Biological sequence processing integration |
| Numba CUDA | GPU kernel acceleration |
| CuPy | GPU-based numerical computation |
| Dask | Distributed task scheduling |
| Textual | Terminal user interface |
| Pytest | Automated testing |
| Git | Version control |

Some technologies are intended for upcoming components and should not be considered fully integrated until implemented and tested.

---

## System Architecture

The planned processing pipeline is:

```text
       Genomic FASTA Dataset
                 |
                 v
          FASTA Parser
                 |
                 v
       Sequence Normalizer
                 |
                 v
         Two-Bit Encoder
                 |
                 v
          Genomic Chunker
                 |
                 v
        Alignment Interface
                 |
          +------+------+
          |             |
          v             v
      CPU Engine    GPU Engine
          |             |
          +------+------+
                 |
                 v
       Mismatch Detection
                 |
                 v
          PAM Evaluation
                 |
                 v
       Off-Target Results
                 |
                 v
        Terminal Dashboard
```

Distributed scheduling using Dask is planned to coordinate processing across multiple chunks and workers.

---

## Core Components

### 1. FASTA Parser

Reads genomic FASTA files and extracts sequence records.

Responsibilities:
- Streaming FASTA file reading.
- Multiple sequence record support.
- Compressed FASTA support.
- Chromosome sequence access.
- Sequence length indexing.

### 2. Sequence Normalizer

Prepares DNA sequences for computational processing.

Responsibilities:
- Converts lowercase bases to uppercase.
- Removes sequence whitespace.
- Converts RNA uracil (U) to DNA thymine (T).
- Validates IUPAC nucleotide symbols.
- Detects ambiguous nucleotide regions.

### 3. Two-Bit Encoder

Encodes DNA bases using two bits per nucleotide.

| DNA Base | Binary Representation |
|---|---|
| A | 00 |
| C | 01 |
| G | 10 |
| T | 11 |

This representation reduces memory requirements compared with storing each nucleotide as a separate byte.

### 4. Genomic Chunker

Divides large genomic sequences into smaller processing units.

Responsibilities:
- Configurable chunk sizes.
- Overlapping sequence regions.
- Global genomic coordinate preservation.
- Chunk metadata management.
- Boundary-aware processing.

Overlapping regions help prevent candidate sequences from being missed at chunk boundaries.

### 5. Alignment Interface

Defines a common interface for sequence alignment engines.

This allows different implementations, such as CPU and GPU aligners, to follow a consistent contract.

### 6. CPU Alignment Engine

Provides the initial reference implementation for sequence matching.

Responsibilities:
- Sliding-window sequence comparison.
- Mismatch counting.
- Mismatch position tracking.
- Bounded batch processing.
- Core-region candidate filtering to avoid duplicate boundary results.

The CPU implementation serves as a correctness baseline for future accelerated implementations.

### 7. GPU Alignment Engine

Planned component using Numba CUDA and CuPy to accelerate genomic sequence comparisons on compatible NVIDIA GPUs.

### 8. Distributed Scheduler

Planned component using Dask to distribute genomic chunks across workers.

### 9. Terminal Dashboard

Planned component using Textual to display processing status, execution metrics, and alignment results.

---

## Project Structure

```text
Gene-Weaver/
│
├── src/
│   └── geneweaver/
│       ├── alignment/
│       ├── sequence_processor/
│       ├── genomic_parser/
│       ├── gpu_engine/
│       ├── scheduler/
│       └── dashboard/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── benchmarks/
│
├── docs/
│
├── data/
│   ├── raw/
│   │   └── GRCh38/
│   ├── processed/
│   └── samples/
│
├── .gitignore
├── pyproject.toml
└── README.md
```

The directory structure represents the intended organization; individual module names may evolve as development progresses.

---

## Installation and Setup

### Prerequisites

- Python 3.12+
- Git
- Windows, Linux, or macOS
- NVIDIA GPU and compatible CUDA environment for actual CUDA execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Gene-Weaver
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install project dependencies

```powershell
pip install -e ".[dev]"
```

This command assumes development dependencies are configured in `pyproject.toml`.

---

## Genomic Dataset

GeneWeaver uses genomic reference sequences in FASTA format.

### Reference Genome

- Organism: Homo sapiens
- Assembly: GRCh38.p14
- Assembly accession: `GCF_000001405.40`
- Chromosome 1 reference accession: `NC_000001.11`

Reference resource:

[NCBI Genome – GRCh38.p14](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/)

### Current Development Dataset

The current local dataset is a selected chromosome 1 region:

| Property | Value |
|---|---|
| File | `chr1_region.fasta.fa` |
| Chromosome | 1 |
| Reference accession | `NC_000001.11` |
| Start coordinate | 85,401,937 |
| End coordinate | 110,297,579 |
| Expected length | 24,895,643 bp |

Place the dataset at:

```text
data/raw/GRCh38/chr1_region.fasta.fa
```

**Important:** Raw genomic datasets are excluded from Git. Download the reference data separately using the NCBI resource above.

---

## Testing

GeneWeaver uses Pytest for automated testing.

Run the complete test suite:

```powershell
pytest -v
```

Run unit tests:

```powershell
pytest tests/unit -v
```

Run a specific test module:

```powershell
pytest tests/unit/test_cpu_aligner.py -v
```

Tests cover sequence parsing, normalization, encoding, chunking, alignment interfaces, and CPU alignment behavior.

---

## Performance and Benchmarking

The project aims to compare:

- CPU alignment performance.
- GPU-accelerated alignment performance.
- Memory consumption.
- Processing throughput.
- Distributed execution performance.

Performance claims will be based on measured benchmarks rather than theoretical estimates.

The long-term performance target is to explore substantial GPU speedup and whole-genome processing within a few minutes. These are development goals, not currently verified results.

---

## Current Development Status

| Component | Status |
|---|---|
| Project architecture and documentation | Completed |
| Project scaffolding | Completed |
| FASTA parser | Implemented |
| Sequence normalizer | Implemented |
| Two-bit encoder | Implemented |
| Genomic chunker | Implemented |
| Alignment interface | Implemented |
| CPU alignment engine | Implemented |
| Reverse-strand alignment | Pending |
| PAM evaluation and scoring | Pending |
| CUDA alignment engine | Pending |
| Dask distributed scheduler | Pending |
| Textual dashboard | Pending |
| Performance benchmarking | Pending |

---

## Scientific Considerations

A sequence match does not automatically indicate that CRISPR-Cas9 will cut that genomic location.

Actual cleavage depends on several biological factors, including:

- Guide RNA sequence.
- PAM compatibility.
- Mismatch positions.
- Cas nuclease specificity.
- Chromatin accessibility and other biological conditions.

GeneWeaver identifies computational candidate off-target sites. Its results should not be interpreted as experimental proof of DNA cleavage.

---

## Future Improvements

- Reverse-complement sequence alignment.
- PAM-aware candidate filtering.
- Mismatch position and seed-region analysis.
- CUDA kernel implementation.
- Distributed genomic processing with Dask.
- Whole-genome reference support.
- Interactive terminal dashboard.
- CPU versus GPU performance benchmarking.
- Improved biological validation and scoring.

---

## Project Purpose

GeneWeaver explores the intersection of bioinformatics, systems programming, parallel computing, and high-performance computing.

The goal is to build a modular genomic alignment engine that can process large DNA sequences efficiently while maintaining correctness, coordinate accuracy, and reproducible performance measurements.