"""Print the published main-body table snapshots; no model or data download."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def render(table):
    records = table["records"]
    if "accuracy_percent" in records[0]:
        headers = ["Target backbone", "Method", "BioASQ", "MedMCQA", "MedQA", "MMLU-Med", "PubMedQA", "Avg."]
        delta = "delta_pp" in records[0]
        if delta:
            headers.append("Delta (pp)")
        rows = []
        for r in records:
            row = [r["target_backbone"], r["method"]]
            row += [f"{v:.2f}" for v in r["accuracy_percent"].values()]
            row += [f"{r['average_percent']:.2f}"]
            if delta:
                row += ["--" if r["delta_pp"] is None else f"{r['delta_pp']:+.2f}"]
            rows.append(row)
    elif "configuration" in records[0]:
        headers = ["Configuration", "LLaMA Avg.", "Delta (pp)", "Qwen Avg.", "Delta (pp)"]
        rows = []
        for r in records:
            row = [r["configuration"]]
            for target in ["LLaMA-3-8B", "Qwen2.5-7B"]:
                value = r[target]
                row += [f"{value['average_percent']:.2f}", "--" if value["delta_pp"] is None else f"{value['delta_pp']:+.2f}"]
            rows.append(row)
    elif "trained_verifier_mean_accuracy_percent" in records[0]:
        headers = ["Target backbone", "Client A", "Client B", "Client C", "Client D", "Client E", "Pool"]
        rows = []
        for r in records:
            diagonal = r["pretrained_diagonal"]
            values = []
            for client, value in r["trained_verifier_mean_accuracy_percent"].items():
                text = f"{value:.2f}"
                if client == diagonal["client"]:
                    text += f" ({diagonal['mean_accuracy_percent']:.2f})"
                values.append(text)
            rows.append([r["target_backbone"], *values, f"{r['pool_mean_accuracy_percent']:.2f}"])
    elif "rule" in records[0]:
        headers = ["Rule", "Uniform LLaMA", "Uniform Qwen", "Heterogeneous LLaMA", "Heterogeneous Qwen"]
        rows = [[r["rule"], *[f"{r[setting][target]:.2f}" for setting in ["uniform", "heterogeneous"] for target in ["LLaMA-3-8B", "Qwen2.5-7B"]]] for r in records]
    else:
        headers = ["Client", "Dataset", "Generator", "Verifier"]
        rows = [[r[k] for k in ["client", "dataset", "generator", "verifier"]] for r in records]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="main_results", help="Table key; use --list to see available keys.")
    parser.add_argument("--list", action="store_true", help="List tables without loading models.")
    parser.add_argument("--json", action="store_true", help="Print a structured table instead of Markdown.")
    args = parser.parse_args()
    data = json.loads((ROOT / "results" / "paper_tables.json").read_text(encoding="utf-8"))
    tables = data["tables"]
    if args.list:
        for name, table in tables.items():
            print(f"{name}: Table {table['table_number']} ({len(table['records'])} rows)")
        return
    if args.table not in tables:
        parser.error("Unknown table. Choose: " + ", ".join(tables))
    table = tables[args.table]
    print(json.dumps(table, indent=2) if args.json else render(table))


if __name__ == "__main__":
    main()
