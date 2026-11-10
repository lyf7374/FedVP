"""Download and export the five FedCoT datasets with the project's fixed splits.

Only data files are downloaded. The five _process_* functions are loaded unchanged
from the checked-out author source, without importing its datasets dependency.
"""
import argparse
import ast
import csv
import io
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
MMLU_SUBJECTS = (
    "clinical_knowledge", "college_biology", "college_medicine",
    "high_school_biology", "medical_genetics", "professional_medicine", "virology",
)
EXPECTED_COUNTS = {
    "pubmedqa": (450, 50, 500), "medmcqa": (2000, 4183, 4183),
    "medqa": (10178, 1272, 1273), "mmlu": (1299, 162, 163),
    "bioasq": (494, 0, 124),
}
HF_BASE = "https://huggingface.co/datasets"
SOURCES = {
    "pubmedqa": {"train": f"{HF_BASE}/qiaojin/PubMedQA/resolve/main/pqa_labeled/train-00000-of-00001.parquet"},
    "medmcqa": {split: f"{HF_BASE}/openlifescienceai/medmcqa/resolve/main/data/{split}-00000-of-00001.parquet"
                for split in ("train", "validation")},
    "medqa": {split: f"{HF_BASE}/bigbio/med_qa/resolve/main/med_qa_en_4options_bigbio_qa/{split}-00000-of-00001.parquet"
              for split in SPLITS},
    "mmlu": {"archive": "https://people.eecs.berkeley.edu/~hendrycks/data.tar"},
}


def configure_environment(root):
    paths = {
        "HF_HOME": root / "cache" / "huggingface",
        "HF_DATASETS_CACHE": root / "cache" / "huggingface" / "datasets",
        "HUGGINGFACE_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "XDG_CACHE_HOME": root / "cache",
        "TMP": root / "cache" / "tmp", "TEMP": root / "cache" / "tmp",
        "TMPDIR": root / "cache" / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def load_author_processors(root=ROOT):
    source = root / "upstream" / "FedCoT" / "src" / "dataset_loader.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {f"_process_{name}_item" for name in EXPECTED_COUNTS}
    selected = [node for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return {name: namespace[f"_process_{name}_item"] for name in EXPECTED_COUNTS}


def shuffled(rows, seed):
    # datasets.Dataset.shuffle(seed=...) uses a fresh default_rng permutation.
    return [rows[index] for index in np.random.default_rng(seed).permutation(len(rows))]


def prepare_records(dataset_name, source, seed=42, processors=None):
    processors = processors if processors is not None else load_author_processors()
    rng = np.random.default_rng(seed)
    python_rng = random.Random(seed)
    if dataset_name == "pubmedqa":
        rows = shuffled(source["train"], seed)
        selected = (rows[:450], rows[450:500], rows[500:1000])
    elif dataset_name == "medmcqa":
        train = [row for row in shuffled(source["train"], seed) if row["exp"] is not None][:2000]
        validation = shuffled(source["validation"], seed)
        selected = (train, validation, validation)
    elif dataset_name == "medqa":
        selected = tuple(shuffled(source[split], seed) for split in SPLITS)
    elif dataset_name == "mmlu":
        rows = [row for subject in MMLU_SUBJECTS
                for split in ("test", "validation", "train") for row in source[subject][split]]
        python_rng.shuffle(rows)
        boundary_80, boundary_90 = int(0.8 * len(rows)), int(0.9 * len(rows))
        selected = (rows[:boundary_80], rows[boundary_80:boundary_90], rows[boundary_90:])
    elif dataset_name == "bioasq":
        rows = list(source)
        python_rng.shuffle(rows)
        boundary = int(0.8 * len(rows))
        selected = (rows[:boundary], [], rows[boundary:])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # Preserve one option-shuffle RNG across train, validation, then test.
    return {split: [processors[dataset_name](row, rng) for row in rows]
            for split, rows in zip(SPLITS, selected)}


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f"Using existing {destination}", flush=True)
        return destination
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "FedCoT-data-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)", flush=True)
    return destination


def read_mmlu(archive):
    result = {}
    # lukaemon/mmlu defines SplitGenerators in this order; train reads dev CSV.
    source_splits = (("test", "test"), ("validation", "val"), ("train", "dev"))
    with tarfile.open(archive, "r:") as tar:
        members = {member.name.removeprefix("./"): member for member in tar.getmembers()}
        for subject in MMLU_SUBJECTS:
            result[subject] = {}
            for split, original in source_splits:
                name = f"data/{original}/{subject}_{original}.csv"
                with tar.extractfile(members[name]) as binary:
                    reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8", newline=""))
                    result[subject][split] = [dict(zip(("input", "A", "B", "C", "D", "target"), row))
                                             for row in reader]
    return result


def read_bioasq(raw_root):
    rows = []
    for year in (11, 10, 9, 8, 7):
        folder = raw_root / f"Task{year}BGoldenEnriched"
        files = sorted(folder.glob(f"{year}B*_golden.json"),
                       key=lambda path: int(path.stem.split("B")[-1].split("_")[0]))
        if not files:
            raise FileNotFoundError(f"BioASQ source files are required under {folder}")
        for path in files:
            content = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(row for row in content["questions"] if row.get("type") == "yesno")
    return rows


def load_sources(root, dataset_name):
    raw_root = root / "data" / "raw" / dataset_name
    if dataset_name == "bioasq":
        return read_bioasq(raw_root), [
            f"https://participants-area.bioasq.org/Tasks/{year}b/goldenDataset/"
            for year in (11, 10, 9, 8, 7)]
    urls = SOURCES[dataset_name]
    if dataset_name == "mmlu":
        archive = download_file(urls["archive"], raw_root / "data.tar")
        return read_mmlu(archive), list(urls.values())
    import pyarrow.parquet as pq
    result = {}
    for split, url in urls.items():
        path = download_file(url, raw_root / f"{split}.parquet")
        result[split] = pq.read_table(path).to_pylist()
    return result, list(urls.values())


def write_processed(output_root, dataset_name, records, sources, seed):
    destination = output_root / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (destination / f"{split}.json").open("w", encoding="utf-8") as stream:
            json.dump(records[split], stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    manifest = {"sources": sources, "counts": {split: len(records[split]) for split in SPLITS},
                "seed": seed, "status": "prepared"}
    with (destination / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="+", choices=tuple(EXPECTED_COUNTS),
                        default=["pubmedqa", "medmcqa", "medqa", "mmlu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    configure_environment(root)
    processors = load_author_processors(root)
    for dataset_name in args.datasets:
        raw, sources = load_sources(root, dataset_name)
        records = prepare_records(dataset_name, raw, args.seed, processors)
        counts = tuple(len(records[split]) for split in SPLITS)
        if counts != EXPECTED_COUNTS[dataset_name]:
            raise ValueError(f"{dataset_name}: expected {EXPECTED_COUNTS[dataset_name]}, got {counts}")
        manifest = write_processed(root / "data" / "processed", dataset_name, records, sources, args.seed)
        print(f"Prepared {dataset_name}: {manifest['counts']}", flush=True)


if __name__ == "__main__":
    main()
