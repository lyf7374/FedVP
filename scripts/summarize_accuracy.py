"""Compute the paper's equal-weight mean from five complete dataset evaluations."""
import argparse
import json
from pathlib import Path

COUNTS = {"bioasq": 124, "medmcqa": 4183, "medqa": 1273, "mmlu": 163, "pubmedqa": 500}

def summarize(files):
    if set(files) != set(COUNTS):
        raise ValueError("All five dataset results are required")
    cells = {}
    for dataset, count in COUNTS.items():
        result = json.loads(Path(files[dataset]).read_text(encoding="utf-8"))
        questions = result.get("questions", result.get("question_count"))
        if result.get("status") != "complete" or questions != count:
            raise ValueError(f"Incomplete {dataset} result")
        correct = result["correct"]
        if type(correct) is not int or not 0 <= correct <= count:
            raise ValueError(f"Invalid correct count for {dataset}")
        cells[dataset] = dict(correct=correct, questions=count, accuracy_percent=100*correct/count)
    return dict(status="complete", datasets=cells,
                mean_accuracy_percent=sum(c["accuracy_percent"] for c in cells.values())/5)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",action="append",required=True,metavar="DATASET=JSON")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    pairs=[value.split("=",1) for value in args.result]
    if len({name for name,path in pairs})!=len(pairs):
        parser.error("Each dataset must appear exactly once")
    result=summarize(dict(pairs))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))

if __name__=="__main__":main()
