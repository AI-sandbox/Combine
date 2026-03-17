# Combine

<p align="left">
  <img src="assets/combine_logo.svg" alt="Combine logo" width="240" />
</p>

Combine implements compressed representations of SNP genotypes and local ancestry (LAI) blocks, allowing matrix-vector products to operate directly on the sparse encoding stored in `.snpdat` files. It supports two ancestry-aware representations:

- **Combine-R**: allele dosages + ancestry dosages per SNP.
- **Combine-S**: ancestry-specific allele dosages + ancestry dosages per SNP.

Group lasso solver used in Combine: [adelie](https://github.com/JamesYang007/adelie/tree/combine).

---

## Installation

```bash
# 1. Clone and install solver with Combine support
cd ..
git clone https://github.com/JamesYang007/adelie/tree/combine
cd adelie
pip install -e .
cd ..

# 2. Clone and install Combine
git clone https://github.com/davidbonet/combine.git
cd Combine
pip install -e .
```

> **Python version**: `>=3.9, <3.11` is required. The build uses pybind11 and compiles C++/Eigen extensions with OpenMP support.

---

## Input files

Combine reads two types of per-chromosome input files: **genotype files** (PLINK 2 PGEN format) and **local ancestry files** (MSP format from tools like [Gnomix](https://github.com/AI-sandbox/gnomix) or [RFMix](https://github.com/slowkoni/rfmix)).

### Genotype files (PGEN / PVAR / PSAM)

Standard PLINK 2 binary genotype files, one set per chromosome:

| File | Description |
|------|-------------|
| `chr{N}.pgen` | Binary genotype matrix (phased haplotypes) |
| `chr{N}.pvar` | Variant info: tab-separated with columns `CHROM`, `POS`, `ID`, `REF`, `ALT` (no header row; comment lines start with `#`) |
| `chr{N}.psam` | Sample info: tab-separated with an `IID` column (integer sample IDs). If the column is named `#IID`, rename it to `IID` before use |

**PSAM formatting**: sample IDs must be plain integers. If they contain suffixes or non-integer characters (e.g. `HG12345_HG12345`, or mixed IDs), strip non-digit characters.

**PVAR formatting**: ensure exactly 5 columns that correspond to (`CHROM  POS  ID  REF  ALT`), with **no header**. Drop any extra trailing columns if present.

### Local ancestry files (MSP)

Tab-separated files output by local ancestry inference tools (e.g. Gnomix, RFMix), one per chromosome. Expected structure:

```
#Subpopulation order/codes: EUR=0	EAS=1	AFR=2	AMR=3	OCE=4	SAS=5
#chm	spos	epos	sgpos	egpos	n snps	<sample_id>.0	<sample_id>.1	...
1	16103	920398	0.02	0.93	550	0	0	2	2	...
```

- **Line 1** (comment): ancestry code mapping.
- **Line 2** (header): starts with `#chm`, followed by window metadata columns and per-haplotype ancestry columns (`<IID>.0` and `<IID>.1` for each sample).
- **Remaining lines**: one row per ancestry window, with integer ancestry codes per haplotype.

**Common MSP fixes** (may be needed after subsetting or exporting):

```bash
# Ensure the header line starts with '#chm' (not 'chm')
sed -i '2s/^chm/#chm/' gnomix_output_chr*.msp

# Ensure a trailing newline (avoids EOF parsing errors)
for chr in {1..22}; do
    echo "" >> gnomix_output_chr${chr}.msp
done
```

---

## Output files (.snpdat)

The `.snpdat` format is a compressed binary representation that encodes genotypes and ancestries together, enabling efficient matrix-vector operations without decompressing. One `.snpdat` file is produced per chromosome and sample split (e.g. `chr1.tr.snpdat`, `chr1.vl.snpdat`).

---

## API reference

### `combine.api.cache_combine_r_snpdat`

Creates a `.snpdat` file in **Combine-R** format.

```python
combine.api.cache_combine_r_snpdat(
    pgen: str,                          # path to .pgen file
    pvar: str,                          # path to .pvar file
    psam: str,                          # path to .psam file
    msp: str,                           # path to .msp file
    dest: str,                          # output .snpdat path
    *,
    sample_indices: np.ndarray = None,  # 0-based sample subset (sorted)
    snp_indices: np.ndarray = None,     # 0-based SNP subset (sorted)
    sort_indices: bool = True,          # auto-sort indices for fast I/O
    n_threads: int = 1,                 # worker threads
)
```

### `combine.api.cache_combine_s_snpdat`

Creates a `.snpdat` file in **Combine-S** format.

```python
combine.api.cache_combine_s_snpdat(
    pgen: str,                          # path to .pgen file
    pvar: str,                          # path to .pvar file
    psam: str,                          # path to .psam file
    msp: str,                           # path to .msp file
    dest: str,                          # output .snpdat path
    *,
    sample_indices: np.ndarray = None,  # 0-based sample subset (sorted)
    snp_indices: np.ndarray = None,     # 0-based SNP subset (sorted)
    sort_indices: bool = True,          # auto-sort indices for fast I/O
    n_threads: int = 1,                 # worker threads
)
```

---

## Usage example

A typical workflow to generate `.snpdat` files from a biobank dataset:

```python
import numpy as np
import pandas as pd
import combine

chromosomes = np.arange(1, 23)
n_threads = 8

pgen_path = lambda c: f"data/pgen/chr{c}.pgen"
pvar_path = lambda c: f"data/pgen/chr{c}.pvar"
psam_path = lambda c: f"data/pgen/chr{c}.psam"
msp_path  = lambda c: f"data/msp/gnomix_output_chr{c}.msp"
snpdat_path = lambda c, mode: f"data/snpdat/chr{c}.{mode}.snpdat"
```

### 1. Verify input coherence

```python
combine.api.check_call_coherence(
    [psam_path(c) for c in chromosomes],
)
combine.api.check_msp_coherence(
    [msp_path(c) for c in chromosomes],
)
for c in chromosomes:
    combine.api.check_call_msp_coherence(
        psam=psam_path(c), msp=msp_path(c),
    )
```

### 2. Filter SNPs by MAF and MSP coverage

```python
import pickle

maf_tol = 0.001
snp_indices_list = []

for c in chromosomes:
    calldata, _, _ = combine.api.read_calldata(
        pgen=pgen_path(c), n_threads=n_threads,
    )
    calldata = combine.ops.calldata_sum(calldata, n_threads=n_threads)
    snp_indices_list.append(
        combine.ops.maf_subset(calldata, maf_tol=maf_tol, n_threads=n_threads)
    )
```

Then prune to SNPs within the MSP window range:

```python
snp_msp_indices_list = []
for i, c in enumerate(chromosomes):
    pvar_df = pd.read_csv(
        pvar_path(c),
        sep='\t',
        names=['CHROM', 'POS', 'ID', 'REF', 'ALT']
    )
    reader = combine.io.MSPReader(msp_path(c))
    reader.read(hap_ids_indices=[0, 1], n_threads=1)

    snps = pvar_df.iloc[snp_indices_list[i]]["POS"]
    in_range = (snps >= reader.pos[0, 0]) & (snps <= reader.pos[-1, -1])
    snp_msp_indices_list.append(snps[in_range].index.to_numpy())
```

### 3. Define sample splits

```python
phe_df = pd.read_csv("data/phenotype.csv")

# Match phenotype IIDs to PSAM row indices
psam_df = pd.read_csv(psam_path(1), sep="\t")
psam_df["psam_index"] = psam_df.index
phe_df = phe_df.merge(psam_df[["IID", "psam_index"]], on="IID", how="left")
phe_df = phe_df.sort_values("psam_index")

tr_indices = np.sort(phe_df[phe_df["split"] == "train"].index)
vl_indices = np.sort(phe_df[phe_df["split"] == "val"].index)
ts_indices = np.sort(phe_df[phe_df["split"] == "test"].index)
```

### 4. Generate `.snpdat` files

```python
modes = ["tr", "vl", "ts"]
split_indices = [tr_indices, vl_indices, ts_indices]

for i, c in enumerate(chromosomes):
    snp_indices = snp_msp_indices_list[i]
    for sample_idx, mode in zip(split_indices, modes):
        psam_indices = phe_df.iloc[sample_idx]["psam_index"].to_numpy().astype(np.uint32)
        
        # Combine-R format
        combine.api.cache_combine_r_snpdat(
            pgen=pgen_path(c),
            pvar=pvar_path(c),
            psam=psam_path(c),
            msp=msp_path(c),
            dest=snpdat_path(c, mode),
            sample_indices=psam_indices,
            snp_indices=snp_indices,
            sort_indices=False,
            n_threads=n_threads,
        )

        # Or Combine-S format
        combine.api.cache_combine_s_snpdat(
            pgen=pgen_path(c),
            pvar=pvar_path(c),
            psam=psam_path(c),
            msp=msp_path(c),
            dest=snpdat_path(c, mode),
            sample_indices=psam_indices,
            snp_indices=snp_indices,
            sort_indices=False,
            n_threads=n_threads,
        )

```

---