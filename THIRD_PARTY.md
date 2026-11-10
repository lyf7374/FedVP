# Sources and dependencies

- **FedCoT:** [DIaacKr/FedCoT](https://github.com/DIaacKr/FedCoT). Data processors and the generation prompt are fetched from revision `5956c1e6c6be2c2b0282b8d7869c282da61e5909` by `scripts/fetch_upstream.py`. Those upstream files are not vendored in this repository. The final-answer-v3 parser preserves the FedCoT answer interface with the extraction corrections used in FedVP. The inspected upstream checkout does not supply a project license file; consult its authors for reuse terms.
- **Models:** exact Hugging Face repository IDs are listed in [the protocol](docs/protocol.md). Download and use them under their respective repository terms, including any required model access agreements. No model weights are bundled here.
- **Datasets:** sources and preparation are documented in [the data guide](docs/data.md). No raw dataset question text is bundled in this release.
- **Baselines:** source repositories and the medical-QA adaptations are listed in [the baseline guide](docs/baselines.md).
- **Runtime:** PyTorch, Transformers, PEFT, Accelerate, SGLang, NumPy, PyArrow and plotting dependencies retain their own licenses.

The project overview image, paper tables, and plotted result values accompany the FedVP manuscript. Code dependencies and citation information are provided separately from model and dataset terms.
