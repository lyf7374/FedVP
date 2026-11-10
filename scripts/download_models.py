"""Download one named Hugging Face model snapshot for a paper generator/verifier."""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.model_id,local_dir=str(args.output))
    print(str(args.output))

if __name__=="__main__":main()
