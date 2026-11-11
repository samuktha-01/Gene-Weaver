# GeneWeaver: End-to-End Data Flow & Pipeline Architecture

## 1. End-to-End Sequence Alignment Data Flow

The end-to-end data flow traces the transformation of raw biological data from external storage through memory buffers, CUDA compute devices, distributed workers, biological scoring, and interactive presentation.

```mermaid
sequenceDiagram
    autonumber
    actor User as Operator / Bioinformatician
    participant TUI as Textual Dashboard
    participant Reader as BioPython FASTA Streamer
    participant Proc as 2-Bit Encoder & Chunker
    participant Dask as Dask Scheduler
    participant Worker as GPU Worker (CUDA)
    participant Scorer as Biological Scoring Engine
    participant Acc as Result Accumulator & Exporter

    User->>TUI: Launch GeneWeaver(FASTA_path, Guide_sequence, Options)
    TUI->>Reader: Initiate Stream Request
    loop For Each Chromosome / Contig
        Reader->>Proc: Stream FASTA Record (String Slices)
        Proc->>Proc: Normalize (Uppercase, IUPAC, N-Filter)
        Proc->>Proc: 2-Bit Encode & Partition Chunks (22bp Halo Overlap)
        Proc->>Dask: Submit Delayed Chunk Task(Chunk_Descriptor, Guide_Bits)
    end
    
    Dask->>Worker: Dispatch Chunk Task to Available GPU Worker
    Worker->>Worker: Allocate Pinned Host Buffer & Async H2D Transfer
    Worker->>Worker: Launch Numba CUDA Sliding-Window Kernel
    Worker->>Worker: Staged Evaluation: PAM Check -> Seed Check -> Distal Check
    Worker->>Worker: Atomic Compaction of Matches (Mismatches <= tau)
    Worker->>Worker: Async D2H Result Transfer to Host
    Worker-->>Dask: Return Raw Candidate Array
    
    Dask-->>Scorer: Stream Raw Candidates (Coordinates, Mismatch Mask)
    Scorer->>Scorer: Extract Exact Base Substitutions
    Scorer->>Scorer: Calculate Hsu-Zhang (2013) Score
    Scorer->>Scorer: Calculate Doench (2016) CFD Score
    Scorer->>Scorer: Evaluate PAM Cleavage Multiplier
    Scorer-->>Acc: Stream Enriched Off-Target Records
    
    Acc->>Acc: Deduplicate Halo Boundary Overlaps
    Acc->>Acc: Update Sorted In-Memory Result Index
    Acc-->>TUI: Post Candidate Event to TUI Reactive Bus
    TUI->>TUI: Render Mismatch Table (Red Highlights) & Telemetry
    
    User->>TUI: Trigger Export Command
    TUI->>Acc: Request Serialization(CSV / Parquet / VCF)
    Acc-->>User: Write Structured Files to Disk
```

---

## 2. Genomic Data Processing Workflow

The diagram below illustrates the exact transformations applied to genomic sequence data before submission to compute devices:

```mermaid
flowchart TD
    A["Raw Genomic FASTA File\n(.fa, .fasta, .fa.gz)"] --> B["BioPython SeqIO.parse Streamer"]
    
    subgraph Sanitization ["Stage 1: Sanitization & Normalization"]
        B --> C["Extract Header & Identifier\n(e.g., '>chr21 GRCh38')"]
        B --> D["Stream Raw ASCII Sequence Chunks"]
        D --> E["Uppercase Normalization\n(a,c,g,t -> A,C,G,T)"]
        E --> F["IUPAC Ambiguity Resolution\n(Mask or Sentinel Coding)"]
        F --> G{"Poly-N Run Detection\n(Length >= 20 bp?)"}
        G -- Yes --> H["Bypass / Skip Segment\n(Omit from GPU Dispatch)"]
        G -- No --> I["Sanitized DNA Character Stream"]
    end

    subgraph Encoding ["Stage 2: 2-Bit Binary Encoding"]
        I --> J["Vectorized 2-Bit Lookup\nA: 00 | C: 01 | G: 10 | T: 11"]
        J --> K["Continuous uint8 Byte Array\n(1 Byte = 1 Nucleotide)"]
        K --> L["Packed uint64 Array\n(32 Nucleotides per Word)"]
    end

    subgraph Chunking ["Stage 3: Chunk Partitioning & Halo Assignment"]
        K --> M["Fixed-Size Windowing\n(Chunk Size C = 10,000,000 bp)"]
        M --> N["Append Halo Margin H = 22 bp\n(Size = C + 22 bp)"]
        N --> O["Attach Genomic Metadata Descriptor\n(Chrom, ChunkIdx, GlobalStart, HaloOffset)"]
        O --> P["Pinned Host Memory Ring Buffer\n(cuda.pinned_array)"]
    end

    P --> Q["Dask Work Queue"]
```

---

## 3. GPU Execution & Memory Transfer Workflow

The GPU pipeline utilizes triple-buffering across CUDA Streams to ensure that Host-to-Device transfer, compute kernel execution, and Device-to-Host result collection run fully concurrently:

```mermaid
flowchart TB
    subgraph Host ["Host Memory (CPU RAM)"]
        BUF0["Pinned Host Buffer 0\n(Chunk N)"]
        BUF1["Pinned Host Buffer 1\n(Chunk N+1)"]
        BUF2["Pinned Host Buffer 2\n(Chunk N+2)"]
        RES_HOST["Host Result Receiver\n(Scoring & Aggregation Queue)"]
    end

    subgraph Streams ["CUDA Asynchronous Execution Pipeline"]
        subgraph StreamA ["CUDA Stream A"]
            H2D_A["Async Memcpy H2D\n(Chunk N)"]
            KERN_A["Numba CUDA Kernel\n(Align Chunk N)"]
            D2H_A["Async Memcpy D2H\n(Results N)"]
            H2D_A --> KERN_A --> D2H_A
        end

        subgraph StreamB ["CUDA Stream B"]
            H2D_B["Async Memcpy H2D\n(Chunk N+1)"]
            KERN_B["Numba CUDA Kernel\n(Align Chunk N+1)"]
            D2H_B["Async Memcpy D2H\n(Results N+1)"]
            H2D_B --> KERN_B --> D2H_B
        end

        subgraph StreamC ["CUDA Stream C"]
            H2D_C["Async Memcpy H2D\n(Chunk N+2)"]
            KERN_C["Numba CUDA Kernel\n(Align Chunk N+2)"]
            D2H_C["Async Memcpy D2H\n(Results N+2)"]
            H2D_C --> KERN_C --> D2H_C
        end
    end

    subgraph DeviceMemory ["GPU Physical VRAM"]
        CONST["Constant Memory Cache (64 KB)\n[Guide Target 23bp + Positional Weights]"]
        D_BUF0["Device Chunk Buffer A"]
        D_BUF1["Device Chunk Buffer B"]
        D_BUF2["Device Chunk Buffer C"]
        D_OUT["Device Candidate Output Ring Buffer\n+ Atomic Counter"]
    end

    BUF0 --> H2D_A --> D_BUF0
    BUF1 --> H2D_B --> D_BUF1
    BUF2 --> H2D_C --> D_BUF2

    CONST -. Broadcast .-> KERN_A
    CONST -. Broadcast .-> KERN_B
    CONST -. Broadcast .-> KERN_C

    D_BUF0 --> KERN_A --> D_OUT
    D_BUF1 --> KERN_B --> D_OUT
    D_BUF2 --> KERN_C --> D_OUT

    D_OUT --> D2H_A --> RES_HOST
    D_OUT --> D2H_B --> RES_HOST
    D_OUT --> D2H_C --> RES_HOST
```

---

## 4. Dask Distributed Processing Architecture

GeneWeaver orchestrates parallel genomic chunks across available GPUs and CPU fallbacks using a dynamic Dask task graph:

```mermaid
flowchart TD
    subgraph Master ["Dask Controller Node"]
        INPUT["Genomic Chunk Streamer"]
        SCHED["Dask Distributed Scheduler\n(Dynamic Work-Stealing)"]
        GRAPH["Delayed Dependency Graph\n(Chunks 1..N)"]
        INPUT --> GRAPH --> SCHED
    end

    subgraph Workers ["Heterogeneous Worker Pool"]
        subgraph GPUWorker0 ["Dask Worker 0 (GPU:0 - RTX 4090)"]
            W0_QUEUE["Task Queue"]
            W0_CUDA["CUDA Context / Stream Engine"]
            W0_QUEUE --> W0_CUDA
        end

        subgraph GPUWorker1 ["Dask Worker 1 (GPU:1 - RTX 4090)"]
            W1_QUEUE["Task Queue"]
            W1_CUDA["CUDA Context / Stream Engine"]
            W1_QUEUE --> W1_CUDA
        end

        subgraph CPUWorker ["Dask Worker 2 (CPU Fallback Worker)"]
            W2_QUEUE["Task Queue"]
            W2_SIMD["NumPy SIMD Thread Pool"]
            W2_QUEUE --> W2_SIMD
        end
    end

    SCHED -->|Assign Chunk 1..k| W0_QUEUE
    SCHED -->|Assign Chunk k+1..2k| W1_QUEUE
    SCHED -->|Assign Chunk 2k+1..3k| W2_QUEUE

    subgraph Collector ["Result Collection & Accumulation"]
        FUTURES["Future Collector / Stream Reduction"]
        W0_CUDA -->|Chunk 1..k Results| FUTURES
        W1_CUDA -->|Chunk k+1..2k Results| FUTURES
        W2_SIMD -->|Chunk 2k+1..3k Results| FUTURES
        FUTURES --> DEDUP["Global Halo Deduplicator"]
        DEDUP --> FINAL["Final Scored Dataset"]
    end
```

---

## 5. Off-Target Detection & Scoring Workflow

The diagram below details the hierarchical filtering and scoring stages applied to identify and rank candidate off-targets:

```mermaid
flowchart TD
    START["Input Candidate 23-mer Window\n(20nt Protospacer + 3nt PAM)"]
    
    subgraph FastReject ["Stage 1: PAM Motif Verification"]
        PAM_CHK{"PAM Sequence Match?\n(Positions 21-23)\nCanonical: NGG\nNon-Canonical: NAG, NGA"}
        PAM_CHK -- Reject --> DROP1["Discard Candidate\n(Zero Memory Overhead)"]
        PAM_CHK -- NGG --> PAM_CANON["PAM Multiplier = 1.00"]
        PAM_CHK -- NAG --> PAM_NAG["PAM Multiplier = 0.25"]
        PAM_CHK -- NGA --> PAM_NGA["PAM Multiplier = 0.10"]
    end

    subgraph MismatchEval ["Stage 2: Mismatch Counting & Seed Verification"]
        PAM_CANON --> SEED_CHK
        PAM_NAG --> SEED_CHK
        PAM_NGA --> SEED_CHK
        SEED_CHK{"Seed Mismatches <= 2?\n(Positions 11-20 proximal to PAM)"}
        SEED_CHK -- No --> DROP2["Discard Candidate\n(Exceeds Seed Tolerance)"]
        SEED_CHK -- Yes --> TOTAL_CHK{"Total Mismatches <= tau?\n(Default tau <= 4 or 5)"}
        TOTAL_CHK -- No --> DROP3["Discard Candidate\n(Exceeds Global Tolerance)"]
        TOTAL_CHK -- Yes --> PASS["Retain Off-Target Candidate Descriptor\n(Offset, Strand, Mismatch Mask)"]
    end

    subgraph BioScoring ["Stage 3: Biological Cleavage Scoring"]
        PASS --> HSU_CALC["Calculate Hsu-Zhang (2013) Score\nS_hsu = product(1 - W[i]) * DistFactor * (1 / m^2)"]
        PASS --> CFD_CALC["Calculate Doench (2016) CFD Score\nS_cfd = PAM_Mult * product(Weight(i, r, d))"]
        HSU_CALC --> COMBINE["Aggregate Scoring Profile\n(Rank by Cleavage Probability)"]
        CFD_CALC --> COMBINE
    end

    subgraph OutputPrep ["Stage 4: Genomic Annotation"]
        COMBINE --> ANNOT["Reconstruct Global Genomic Coordinate\nChrom, Start_bp, End_bp, Strand (+/-)"]
        ANNOT --> MISMATCH_MAP["Map Specific Mismatch Base Substitutions\n(e.g., Pos 14: C -> T)"]
        MISMATCH_MAP --> READY["OffTargetCandidate Record\n(Ready for TUI and Disk Export)"]
    end
```

---

## 6. Data Ingestion & Transformation Lifecycle

The table below outlines the precise data transformation pipeline across each tier of the system:

| Pipeline Step | Input Data Representation | Processing Entity | Output Data Representation | Memory Domain |
| :--- | :--- | :--- | :--- | :--- |
| **1. File Ingestion** | Plain text FASTA (`.fa` / `.fa.gz`) | BioPython `SeqIO.parse` | In-memory Python `SeqRecord` slices | Host OS Buffer $\rightarrow$ Python Heap |
| **2. Sanitization** | Raw ASCII sequence string | `SequenceNormalizer` | Uppercased string, `N`-masked | Python Heap |
| **3. Encoding** | Sanitized ASCII string | `TwoBitEncoder` | 1D `numpy.ndarray` (`uint8` / packed `uint64`) | Host Pageable RAM |
| **4. Chunking & Halo** | Contiguous 2-bit array | `GenomicChunker` | Chunk descriptors with $22\text{ bp}$ halo margin | Pinned Host RAM (`cuda.pinned_array`) |
| **5. H2D Transfer** | Pinned Host Buffer | CUDA Stream `cuda.memcpy_async` | Device VRAM array pointer | GPU Global VRAM |
| **6. Kernel Alignment** | Device sequence buffer + Constant guide | Numba CUDA sliding kernel | Candidate integer descriptors `(idx, strand, mask)` | GPU Shared $\rightarrow$ Global VRAM |
| **7. D2H Transfer** | Device candidate buffer | CUDA Stream `cuda.memcpy_async` | Host NumPy array of candidate hits | GPU VRAM $\rightarrow$ Pinned Host RAM |
| **8. Scoring** | Candidate integer descriptors | `BiologicalScorer` | Typed `OffTargetCandidate` dataclass records | Python Heap |
| **9. TUI Display** | Typed candidate records | Textual TUI Widget | Colored Rich Text Markup (Red Mismatches) | Terminal Emulator Video Buffer |
| **10. Export** | Aggregated candidate records | `ResultExporter` | CSV, Parquet, or VCF files | Disk Storage (NVMe) |

---

## 7. State Transition Lifecycle of a Genomic Chunk

A single genomic chunk progresses through a strict, deterministic state machine:

```
[PENDING]
    |
    v (Submitted to Dask Task Graph)
[QUEUED]
    |
    v (Assigned to GPU Worker; Memory Pinned)
[IN_TRANSFER_H2D]
    |
    v (VRAM Populated; Stream Kernel Launched)
[IN_KERNEL_COMPUTE]
    |
    v (Kernel Exited; Atomic Compaction Complete)
[IN_TRANSFER_D2H]
    |
    v (Host Buffer Populated)
[BIOLOGICAL_SCORING]
    |
    v (Hsu-Zhang & CFD Calculated; Halo Deduplicated)
[AGGREGATED]
```

### State Error Handling & Recovery Rules
1. **Host-to-Device Transfer Timeout**: If PCIe transfer fails or stalls $> 10\text{ s}$, the worker resets the CUDA stream and restarts transfer.
2. **CUDA Kernel Error (Illegal Memory Access / Hardware Fault)**:
   * The Dask worker intercepts the `CudaSupportError`.
   * The worker context is destroyed and recreated.
   * The chunk transitions back to `QUEUED` with an incremented retry counter.
   * If retries exceed 3, the chunk falls back to the CPU worker.
3. **Out-of-Memory (VRAM OOM)**:
   * Dynamic chunk subdivider halves the chunk size ($10\text{ Mbp} \rightarrow 5\text{ Mbp}$) and resubmits the sub-chunks to `QUEUED`.
