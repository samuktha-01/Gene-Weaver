# GeneWeaver: Component Specifications & Interface Contracts

## 1. Component Implementation Status Matrix

This matrix establishes the implementation baseline for GeneWeaver. As identified during repository inspection, the repository is currently in an **uninitialized / greenfield state** (0 commits, 0 source files). Consequently, all components detailed below represent the **proposed production architectural specification**.

| Component / Subsystem | Package Location | Status | Implementation Responsibility | Key Technologies |
| :--- | :--- | :--- | :--- | :--- |
| **FASTA Reader & Validator** | `geneweaver.genomic_parser` | Proposed | Streaming ingestion, index offsets, format validation | BioPython `SeqIO`, gzip |
| **Genomic Coordinate Indexer** | `geneweaver.genomic_parser` | Proposed | Chromosome offset tracking, boundary coordinates | Python, NumPy |
| **Sequence Normalizer** | `geneweaver.sequence_processor` | Proposed | Case folding, IUPAC resolution, Poly-N masking | Python, regex |
| **2-Bit Encoder & Packer** | `geneweaver.sequence_processor` | Proposed | 2-bit binary transformation, uint64 bit-packing | NumPy, bitwise ops |
| **Genomic Chunker with Halo** | `geneweaver.sequence_processor` | Proposed | Overlapping halo slicing, coordinate metadata attachment | NumPy |
| **CPU Baseline & Fallback Aligner** | `geneweaver.alignment` | Proposed | Stride-trick sliding window, Hamming distance baseline | NumPy SIMD |
| **Reverse Complement Generator** | `geneweaver.alignment` | Proposed | Both-strand PAM alignment coordination | NumPy, bitwise ops |
| **CUDA Device & Memory Manager** | `geneweaver.cuda_engine` | Proposed | Device capability queries, pinned memory allocation | Numba CUDA, CuPy |
| **Asynchronous Stream Pipeline** | `geneweaver.cuda_engine` | Proposed | Multi-stream double-buffering (H2D, Compute, D2H) | Numba CUDA Streams |
| **Numba CUDA Alignment Kernels** | `geneweaver.cuda_engine` | Proposed | Constant-memory sgRNA, shared-memory tile, atomic compaction | Numba CUDA JIT |
| **CuPy Vectorized Operations** | `geneweaver.cuda_engine` | Proposed | Device array allocation, prefix sums, filtering | CuPy |
| **Dask Cluster Manager** | `geneweaver.distributed_scheduler`| Proposed | Worker lifecycle, GPU device binding (`CUDA_VISIBLE_DEVICES`) | Dask Distributed |
| **Dynamic Task Graph** | `geneweaver.distributed_scheduler`| Proposed | Chunk task generation, failure recovery, backpressure | Dask Delayed |
| **Biological Scoring Engine** | `geneweaver.scoring` | Proposed | Hsu-Zhang (2013) & CFD (2016) cleavage probabilities | NumPy, SciPy |
| **Result Accumulator & Deduplicator**| `geneweaver.result_manager` | Proposed | Halo boundary deduplication, global top-K candidate sorting | Python, Pandas |
| **Multi-Format Result Exporter** | `geneweaver.result_manager` | Proposed | High-throughput serialization to CSV, Parquet, VCF, JSON | PyArrow, Pandas |
| **Telemetry Collector** | `geneweaver.monitoring` | Proposed | Real-time NVML GPU stats, psutil CPU/RAM metrics | `pynvml`, `psutil` |
| **Textual Dashboard Application** | `geneweaver.dashboard` | Proposed | Async TUI, DNA mismatch rendering, reactive progress | Textual, Rich |
| **Configuration & Hardware Detect** | `geneweaver.configuration` | Proposed | Pydantic v2 settings, CUDA runtime validation | Pydantic, YAML |
| **Automated Benchmarking Suite** | `benchmarks` | Proposed | 100x speedup validation, memory profiling, SLA verification | Pytest, Nsight |

---

## 2. Package Specifications & Interface Contracts

### 2.1 Package: `geneweaver.genomic_parser`

#### Module: `fasta_reader.py`
* **Purpose**: Ingests multi-gigabyte FASTA files via streaming generators without loading entire files into memory.
* **Interface Contract**:
  ```python
  from pathlib import Path
  from typing import Generator, Tuple

  class FastaReader:
      def __init__(self, file_path: Path, buffer_size_kb: int = 64):
          ...
      def stream_chromosomes(self) -> Generator[Tuple[str, Generator[str, None, None]], None, None]:
          """Yields chromosome identifier and an inner generator yielding sequence string slices."""
          ...
      def get_chromosome_lengths(self) -> dict[str, int]:
          """Extracts sequence lengths quickly using index seek tables."""
          ...
  ```
* **Invariants**: Peak RAM consumption remains $< 100\text{ MB}$ regardless of FASTA file size ($> 30\text{ GB}$).

#### Module: `validator.py`
* **Purpose**: Enforces sequence integrity, character validity, and CRISPR guide specification invariants.
* **Interface Contract**:
  ```python
  class SequenceValidator:
      @staticmethod
      def validate_guide(guide: str, expected_len: int = 23) -> str:
          """Validates 20nt spacer + 3nt PAM. Converts RNA 'U' to DNA 'T'. Enforces IUPAC."""
          ...
      @staticmethod
      def validate_fasta_path(path: Path) -> Path:
          """Verifies file existence, permissions, and non-empty size."""
          ...
  ```

---

### 2.2 Package: `geneweaver.sequence_processor`

#### Module: `normalizer.py`
* **Purpose**: Sanitizes raw genomic strings, handles IUPAC degenerate codes, and detects masked unsequenced regions (`N` runs).
* **Interface Contract**:
  ```python
  class SequenceNormalizer:
      def normalize(self, sequence: str) -> str:
          """Uppercases sequence, converts non-standard characters."""
          ...
      def find_poly_n_regions(self, sequence: str, min_length: int = 20) -> list[tuple[int, int]]:
          """Returns list of (start, end) offsets of undetermined base blocks to bypass."""
          ...
  ```

#### Module: `encoder.py`
* **Purpose**: Compresses ASCII nucleotides (`A, C, G, T`) into 2-bit binary integer codes ($00_2, 01_2, 10_2, 11_2$).
* **Interface Contract**:
  ```python
  import numpy as np

  class TwoBitEncoder:
      @staticmethod
      def encode(sequence: str) -> np.ndarray:
          """Converts ASCII string to 1D uint8 array with values in {0, 1, 2, 3}."""
          ...
      @staticmethod
      def decode(bits: np.ndarray) -> str:
          """Reconstructs ASCII nucleotide sequence from 2-bit array."""
          ...
      @staticmethod
      def pack_uint64(encoded_uint8: np.ndarray) -> np.ndarray:
          """Packs 32 2-bit nucleotides into single 64-bit unsigned integers."""
          ...
  ```

#### Module: `chunker.py`
* **Purpose**: Partitions continuous 2-bit genomic arrays into overlapping chunk tiles with global coordinate descriptors.
* **Data Class & Interface Contract**:
  ```python
  from dataclasses import dataclass
  import numpy as np

  @dataclass(frozen=True)
  class ChunkDescriptor:
      chromosome: str
      chunk_index: int
      global_start_bp: int
      nominal_length: int
      halo_length: int
      sequence_data: np.ndarray  # uint8 2-bit array of length (nominal + halo)

  class GenomicChunker:
      def __init__(self, chunk_size: int = 10_000_000, halo_size: int = 22):
          self.chunk_size = chunk_size
          self.halo_size = halo_size

      def partition(self, chromosome: str, sequence: np.ndarray) -> list[ChunkDescriptor]:
          """Generates overlapping chunks with exact halo margins."""
          ...
  ```

---

### 2.3 Package: `geneweaver.alignment`

#### Module: `base.py`
* **Purpose**: Declares the formal aligner protocol and data exchange records.
* **Interface Contract**:
  ```python
  from typing import Protocol
  from dataclasses import dataclass
  import numpy as np

  @dataclass
  class RawCandidateHit:
      chunk_index: int
      local_offset: int
      strand: int          # 0 for forward (+), 1 for reverse complement (-)
      mismatch_count: int
      mismatch_bitmask: int # Bitmask indicating exact positions of mismatches

  class SequenceAligner(Protocol):
      def align(self, chunk: ChunkDescriptor, guide_target: np.ndarray, max_mismatches: int) -> list[RawCandidateHit]:
          ...
  ```

#### Module: `cpu_aligner.py`
* **Purpose**: Implements the CPU baseline and automatic fallback alignment engine using vectorized NumPy strided windows.
* **Technical Invariant**: Must yield the exact same candidate list as the CUDA engine for any given sequence and threshold.

---

### 2.4 Package: `geneweaver.cuda_engine`

#### Module: `device_manager.py`
* **Purpose**: Detects CUDA availability, probes GPU compute capability, and manages device contexts.
* **Interface Contract**:
  ```python
  class CudaDeviceManager:
      @staticmethod
      def is_cuda_available() -> bool:
          ...
      @staticmethod
      def get_device_properties(device_id: int = 0) -> dict:
          """Returns device name, compute capability, total VRAM, SM count."""
          ...
  ```

#### Module: `memory_pool.py`
* **Purpose**: Manages pinned host memory buffers (`cuda.pinned_array`) and CuPy device memory pools to eliminate allocation overhead during execution.

#### Module: `stream_pipeline.py`
* **Purpose**: Coordinates concurrent CUDA streams for asynchronous double-buffered or triple-buffered execution.

#### Module: `kernels.py`
* **Purpose**: Holds the JIT-compiled Numba CUDA kernel specifications.
* **Kernel Function Signature Specification**:
  ```python
  from numba import cuda

  @cuda.jit
  def sliding_window_crispr_kernel(
      d_genomic_chunk,     # Device array: uint8 2-bit encoded sequence
      chunk_length,        # int32: Total chunk length including halo
      d_guide_target,      # Device array: uint8 2-bit target guide (23 bp)
      max_mismatches,      # int32: Cutoff threshold (e.g., 4 or 5)
      d_candidate_indices, # Output array: int32 indices of candidate hits
      d_candidate_strands, # Output array: int8 strand (0: +, 1: -)
      d_mismatch_counts,   # Output array: int8 mismatch counts
      d_mismatch_masks,    # Output array: int32 bitmasks of mismatch locations
      d_global_counter     # Device array (length 1): Atomic counter for result compaction
  ):
      """Massively parallel sliding window kernel with shared memory halo caching."""
      ...
  ```

---

### 2.5 Package: `geneweaver.distributed_scheduler`

#### Module: `cluster_manager.py`
* **Purpose**: Initializes and manages the Dask Distributed client and assigns dedicated worker processes to physical GPUs.
* **Interface Contract**:
  ```python
  from dask.distributed import Client

  class DaskClusterManager:
      def __init__(self, n_gpus: int | None = None, enable_cpu_fallback: bool = True):
          ...
      def start_cluster(self) -> Client:
          """Spawns local Dask cluster with GPU device bindings."""
          ...
      def shutdown(self) -> None:
          ...
  ```

#### Module: `task_graph.py` & `aggregator.py`
* **Purpose**: Builds delayed task dependency graphs for chunks, manages worker queues, and streams completed futures to the biological scoring layer.

---

### 2.6 Package: `geneweaver.scoring`

#### Module: `pam_evaluator.py`
* **Purpose**: Validates PAM sequence compatibility and calculates cleavage factor penalties for canonical (`NGG` = 1.0) and non-canonical (`NAG` = 0.25, `NGA` = 0.10) PAM variants.

#### Module: `hsu_zhang.py`
* **Purpose**: Calculates the Hsu-Zhang (Nature Biotech 2013) score using positional weight penalties and pairwise mismatch distance factors.

#### Module: `cfd_score.py`
* **Purpose**: Evaluates the Doench (Nature Biotech 2016) Cutting Frequency Determination matrix for specific nucleotide substitutions.

#### Module: `ranker.py`
* **Data Class & Interface Contract**:
  ```python
  from dataclasses import dataclass

  @dataclass(frozen=True)
  class OffTargetCandidate:
      chromosome: str
      genomic_start: int
      genomic_end: int
      strand: str              # '+' or '-'
      target_sequence: str     # 23 bp DNA
      guide_sequence: str      # 23 bp DNA
      pam_sequence: str        # 3 bp PAM
      mismatches: int          # Total mismatch count
      mismatch_positions: list[int] # 1-based positions along 20nt spacer
      hsu_score: float         # 0.0 to 100.0
      cfd_score: float         # 0.0 to 1.0
      is_canonical_pam: bool

  class OffTargetRanker:
      def rank(self, candidates: list[OffTargetCandidate], sort_by: str = "cfd") -> list[OffTargetCandidate]:
          """Sorts off-targets by biological hazard score descending."""
          ...
  ```

---

### 2.7 Package: `geneweaver.result_manager`

#### Module: `accumulator.py`
* **Purpose**: Thread-safe aggregation of off-target candidates streamed from Dask workers.
* **Boundary Deduplication**: Removes duplicate candidate hits detected in chunk halo regions using a unique `(chromosome, genomic_start, strand)` composite key.

#### Module: `exporter.py`
* **Purpose**: Serializes off-target candidates into standard bioinformatic data formats:
  - **CSV / TSV**: Readable tabular format with complete metadata.
  - **Apache Parquet**: Columnar compressed format for high-speed downstream analytics in DuckDB or Pandas.
  - **Genomic VCF / BED**: Standard tracks for viewing in IGV (Integrative Genomics Viewer).

---

### 2.8 Package: `geneweaver.monitoring` & `geneweaver.dashboard`

#### Module: `telemetry.py`
* **Purpose**: Background daemon thread that polls hardware telemetry:
  - GPU Core Utilization (%) via NVML (`nvmlDeviceGetUtilizationRates`).
  - GPU VRAM Utilization (Used / Total MB) via NVML (`nvmlDeviceGetMemoryInfo`).
  - CPU Utilization (%) and Host RAM (MB) via `psutil`.

#### Module: `dashboard/app.py`
* **Purpose**: Textual Application providing the interactive terminal UI.
* **Component Widgets**:
  - `GpuMeter`: Displays GPU core and VRAM gauges.
  - `GenomicProgressBar`: Tracks chunks completed vs pending with throughput (Mbp/s) and ETA.
  - `MismatchTable`: Displays ranked off-targets. DNA mismatches are styled using Rich markup:
    ```
    Guide:  G A G T C C G A G C A G A A G A A G A A - A G G
    Target: G A [bold red]A[/] T C C G A G C A G A A G A A G [bold red]C[/] A A - C G G
    ```
  - `LogPanel`: Real-time operational diagnostics and warning alerts.

---

### 2.9 Package: `geneweaver.configuration`

#### Module: `settings.py`
* **Purpose**: Centralized application configuration defined via Pydantic v2:
  ```python
  from pydantic import BaseModel, Field
  from pathlib import Path

  class GeneWeaverSettings(BaseModel):
      fasta_path: Path
      guide_sequence: str = Field(..., min_length=23, max_length=23)
      max_mismatches: int = Field(default=4, ge=0, le=10)
      chunk_size_bp: int = Field(default=10_000_000, ge=100_000)
      halo_size_bp: int = Field(default=22, const=True)
      enable_gpu: bool = True
      gpu_device_ids: list[int] = Field(default_factory=lambda: [0])
      enable_cpu_fallback: bool = True
      scoring_model: str = "cfd"  # 'cfd' | 'hsu' | 'both'
      output_directory: Path = Path("./results")
      tui_refresh_hz: float = 10.0
  ```
