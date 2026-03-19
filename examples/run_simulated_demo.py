"""Format the data for Combine on simulated demo dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import adelie as ad
import numpy as np
import pandas as pd

import combine


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "examples" / "simulated_data"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "simulated_demo_snpdat"
CHROMOSOMES = np.array([1, 2], dtype=np.int32)


def pgen_path(chrom: int) -> Path:
    return DATA_DIR / "pgen" / f"chr{chrom}.pgen"


def pvar_path(chrom: int) -> Path:
    return DATA_DIR / "pgen" / f"chr{chrom}.pvar"


def psam_path(chrom: int) -> Path:
    return DATA_DIR / "pgen" / f"chr{chrom}.psam"


def msp_path(chrom: int) -> Path:
    return DATA_DIR / "msp" / f"gnomix_output_chr{chrom}.msp"


def snpdat_path(out_dir: Path, chrom: int, split_label: str, representation: str) -> Path:
    return out_dir / f"chr{chrom}.{split_label}.{representation}.snpdat"


def filter_snp_indices(chrom: int, n_threads: int, maf_tol: float) -> np.ndarray:
    calldata, _, _ = combine.api.read_calldata(
        pgen=str(pgen_path(chrom)),
        n_threads=n_threads,
    )
    calldata_sum = combine.ops.calldata_sum(calldata, n_threads=n_threads)
    snp_indices = combine.ops.maf_subset(
        calldata_sum,
        maf_tol=maf_tol,
        n_threads=n_threads,
    )

    pvar_df = pd.read_csv(
        pvar_path(chrom),
        sep="\t",
        names=["CHROM", "POS", "ID", "REF", "ALT"],
    )
    reader = combine.io.MSPReader(str(msp_path(chrom)))
    reader.read(hap_ids_indices=[0, 1], n_threads=1)

    snps = pvar_df.iloc[snp_indices]["POS"]
    in_range = (snps >= reader.pos[0, 0]) & (snps <= reader.pos[-1, -1])
    return snps[in_range].index.to_numpy(dtype=np.uint32)


def load_split_indices() -> dict[str, np.ndarray]:
    phenotype_df = pd.read_csv(DATA_DIR / "phenotype.csv")
    psam_df = pd.read_csv(psam_path(1), sep="\t")
    psam_df["psam_index"] = psam_df.index

    phenotype_df = phenotype_df.merge(psam_df[["IID", "psam_index"]], on="IID", how="left")
    phenotype_df = phenotype_df.sort_values("psam_index").reset_index(drop=True)

    return {
        "tr": np.sort(
            phenotype_df.loc[phenotype_df["split"] == "train", "psam_index"].to_numpy(dtype=np.uint32)
        ),
        "vl": np.sort(
            phenotype_df.loc[phenotype_df["split"] == "val", "psam_index"].to_numpy(dtype=np.uint32)
        ),
        "ts": np.sort(
            phenotype_df.loc[phenotype_df["split"] == "test", "psam_index"].to_numpy(dtype=np.uint32)
        ),
    }


def validate_inputs(*, n_threads: int, maf_tol: float) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    combine.api.check_call_coherence([str(psam_path(chrom)) for chrom in CHROMOSOMES])
    combine.api.check_msp_coherence([str(msp_path(chrom)) for chrom in CHROMOSOMES])
    for chrom in CHROMOSOMES:
        combine.api.check_call_msp_coherence(
            psam=str(psam_path(chrom)),
            msp=str(msp_path(chrom)),
        )

    snp_indices_by_chrom = {
        int(chrom): filter_snp_indices(int(chrom), n_threads=n_threads, maf_tol=maf_tol)
        for chrom in CHROMOSOMES
    }
    split_indices = load_split_indices()
    return snp_indices_by_chrom, split_indices


def ensure_combine_support() -> None:
    missing = [
        name
        for name in ("snp_combine_r", "snp_combine_s")
        if not hasattr(ad.io, name)
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "The installed adelie build does not expose the Combine snpdat writers "
            f"({missing_str}). Install the Combine-support adelie build referenced in "
            "the README before running the full demo. "
            f"Resolved Python: {sys.executable}. "
            f"Resolved adelie: {ad.__file__}."
        )


def load_subset_arrays(
    *,
    chrom: int,
    sample_indices: np.ndarray,
    snp_indices: np.ndarray,
    n_threads: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    pvar_df = pd.read_csv(
        pvar_path(chrom),
        sep="\t",
        names=["CHROM", "POS", "ID", "REF", "ALT"],
    )
    lai, _, reader = combine.api.read_lai(
        str(msp_path(chrom)),
        pvar_df["POS"].iloc[snp_indices].to_numpy(),
        sample_indices=sample_indices,
        sort_indices=False,
        n_threads=n_threads,
    )
    calldata, _, _ = combine.api.read_calldata(
        str(pgen_path(chrom)),
        sample_indices=sample_indices,
        snp_indices=snp_indices,
        sort_indices=False,
        n_threads=n_threads,
    )
    return calldata, lai, len(reader.ancestry_map)


def expected_combine_r_dense(calldata: np.ndarray, lai: np.ndarray, n_ancestries: int, n_threads: int) -> np.ndarray:
    n_samples, double_snps = calldata.shape
    n_snps = double_snps // 2
    calldata_sum = combine.ops.calldata_sum(calldata, n_threads=n_threads)
    lai_haps = lai.reshape(n_samples, n_snps, 2)

    expected = np.zeros((n_samples, n_snps * (1 + n_ancestries)), dtype=np.int8)
    expected[:, ::(1 + n_ancestries)] = calldata_sum

    for ancestry in range(n_ancestries):
        dosage = (lai_haps[:, :, 0] == ancestry).astype(np.int8) + (lai_haps[:, :, 1] == ancestry).astype(np.int8)
        expected[:, ancestry + 1::(1 + n_ancestries)] = dosage

    return expected


def expected_combine_s_dense(calldata: np.ndarray, lai: np.ndarray, n_ancestries: int) -> np.ndarray:
    n_samples, double_snps = calldata.shape
    n_snps = double_snps // 2
    calldata_haps = calldata.reshape(n_samples, n_snps, 2)
    lai_haps = lai.reshape(n_samples, n_snps, 2)

    expected = np.zeros((n_samples, n_snps * (2 * n_ancestries)), dtype=np.int8)
    for ancestry in range(n_ancestries):
        mutated = (
            ((calldata_haps[:, :, 0] == 1) & (lai_haps[:, :, 0] == ancestry)).astype(np.int8)
            + ((calldata_haps[:, :, 1] == 1) & (lai_haps[:, :, 1] == ancestry)).astype(np.int8)
        )
        dosage = (lai_haps[:, :, 0] == ancestry).astype(np.int8) + (lai_haps[:, :, 1] == ancestry).astype(np.int8)
        expected[:, ancestry::(2 * n_ancestries)] = mutated
        expected[:, n_ancestries + ancestry::(2 * n_ancestries)] = dosage

    return expected


def verify_combine_r(
    *,
    dest: Path,
    expected_dense: np.ndarray,
    n_samples: int,
    n_snps: int,
    n_ancestries: int,
    n_threads: int,
) -> None:
    handler = ad.io.snp_combine_r(str(dest), read_mode="mmap")
    handler.read()
    assert handler.rows == n_samples
    assert handler.snps == n_snps
    assert handler.ancestries == n_ancestries
    assert handler.cols == n_snps * (1 + n_ancestries)

    dense = handler.to_dense(n_threads=n_threads)
    expected_nnz = np.sum(expected_dense != 0, axis=0)
    np.testing.assert_array_equal(dense, expected_dense)
    np.testing.assert_array_equal(handler.nnz, expected_nnz)

def verify_combine_s(
    *,
    dest: Path,
    expected_dense: np.ndarray,
    n_samples: int,
    n_snps: int,
    n_ancestries: int,
    n_threads: int,
) -> None:
    handler = ad.io.snp_combine_s(str(dest), read_mode="mmap")
    handler.read()
    assert handler.rows == n_samples
    assert handler.snps == n_snps
    assert handler.ancestries == n_ancestries
    assert handler.cols == n_snps * (2 * n_ancestries)

    dense = handler.to_dense(n_threads=n_threads)
    expected_nnz = np.sum(expected_dense != 0, axis=0)
    np.testing.assert_array_equal(dense, expected_dense)
    np.testing.assert_array_equal(handler.nnz, expected_nnz)

def run_demo(
    *,
    out_dir: Path,
    n_threads: int = 1,
    maf_tol: float = 0.05,
) -> None:
    snp_indices_by_chrom, split_indices = validate_inputs(
        n_threads=n_threads,
        maf_tol=maf_tol,
    )

    ensure_combine_support()
    out_dir.mkdir(parents=True, exist_ok=True)

    for chrom in CHROMOSOMES:
        current_snp_indices = snp_indices_by_chrom[int(chrom)]
        for split_label, sample_indices in split_indices.items():
            calldata, lai, n_ancestries = load_subset_arrays(
                chrom=int(chrom),
                sample_indices=sample_indices,
                snp_indices=current_snp_indices,
                n_threads=n_threads,
            )
            n_samples = sample_indices.size
            n_snps = current_snp_indices.size

            dest_r = snpdat_path(out_dir, int(chrom), split_label, "combine_r")
            combine.api.cache_combine_r_snpdat(
                pgen=str(pgen_path(int(chrom))),
                pvar=str(pvar_path(int(chrom))),
                psam=str(psam_path(int(chrom))),
                msp=str(msp_path(int(chrom))),
                dest=str(dest_r),
                sample_indices=sample_indices,
                snp_indices=current_snp_indices,
                sort_indices=False,
                n_threads=n_threads,
            )
            verify_combine_r(
                dest=dest_r,
                expected_dense=expected_combine_r_dense(calldata, lai, n_ancestries, n_threads),
                n_samples=n_samples,
                n_snps=n_snps,
                n_ancestries=n_ancestries,
                n_threads=n_threads,
            )

            dest_s = snpdat_path(out_dir, int(chrom), split_label, "combine_s")
            combine.api.cache_combine_s_snpdat(
                pgen=str(pgen_path(int(chrom))),
                pvar=str(pvar_path(int(chrom))),
                psam=str(psam_path(int(chrom))),
                msp=str(msp_path(int(chrom))),
                dest=str(dest_s),
                sample_indices=sample_indices,
                snp_indices=current_snp_indices,
                sort_indices=False,
                n_threads=n_threads,
            )
            verify_combine_s(
                dest=dest_s,
                expected_dense=expected_combine_s_dense(calldata, lai, n_ancestries),
                n_samples=n_samples,
                n_snps=n_snps,
                n_ancestries=n_ancestries,
                n_threads=n_threads,
            )

    print(f"Wrote demo .snpdat files to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where demo .snpdat outputs will be written.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Thread count passed to Combine I/O helpers.",
    )
    parser.add_argument(
        "--maf-tol",
        type=float,
        default=0.05,
        help="MAF threshold used to retain demo SNPs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(
        out_dir=args.out_dir,
        n_threads=args.n_threads,
        maf_tol=args.maf_tol,
    )


if __name__ == "__main__":
    main()
