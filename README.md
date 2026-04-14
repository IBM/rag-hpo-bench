<!-- This should be the location of the title of the repository, normally the short name -->
# RAG HPO Bench

<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->


## Overview

Exploring hyperparameter optimization (HPO) for RAG systems is expensive, as running experiments across multiple configurations and datasets requires significant computational resources. **RAG-HPO-bench** solves this problem by providing pre-computed results for 162 RAG configurations, enabling researchers to develop and test new HPO algorithms without the computational overhead.

This repository accompanies our paper [An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation](https://arxiv.org/abs/2505.03452) and provides:

- 📊 **[RAG-HPO-bench dataset](https://huggingface.co/datasets/matanor/rag-hpo-bench)**: Pre-computed RAG evaluation results across 162 parameter combinations
- 🔬 **Analysis tools**: Code examples for exploring HPO techniques and finding optimal configurations
- 🚀 **Quick experimentation**: Test new HPO algorithms without expensive RAG pipeline runs

**Use this benchmark to:**
- Develop novel HPO algorithms for RAG systems
- Compare optimization strategies across different datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/IBM/rag-hpo-bench.git
cd rag-hpo-bench

# Install the package
uv pip install -e .
```

**Note:** `uv` will automatically create and use a virtual environment if one doesn't exist. For manual virtual environment setup and other installation options, see [INSTALL.md](INSTALL.md).

## Usage

### HPO

To run HPO experiments with the RAG-HPO-bench dataset:

1. **Run HPO experiments**:
   ```bash
   uv run python -m rag_hpo_bench.hpo.run_hpo_experiments
   ```

   This will run multiple HPO experiments with different combinations of:
   - **Datasets**: ClapNQ, AIArxiv (with Dev/Test splits)
   - **Algorithms**: Grid search, Random search, Greedy-M, Greedy-R
   - **Optimization metrics**: LLMaaJ-AC, Lexical-AC, Lexical-FF

2. **Limit the number of experiments** (optional):
   ```bash
   uv run python -m rag_hpo_bench.hpo.run_hpo_experiments --max-experiments 5
   ```

Results will be saved to `./experiments_output/` with subdirectories organized by algorithm, dataset, and metric.

### Other Usage Examples

For additional usage examples, see the [examples/README.md](examples/README.md) which includes scripts for:
- Finding best configurations across datasets
- Analyzing RAG configuration results

## Notes

If you have any questions or issues you can create a new [issue here][issues].

Pull requests are very welcome! Make sure your patches are well tested.
Ideally create a topic branch for every separate change you make. For
example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## Citation

Please cite the paper if you use the RAG-HPO-bench dataset or code:

```bibtex
@article{orbach2025raghpo,
  title={An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation},
  author={Orbach, Matan and Eytan, Ohad and Sznajder, Benjamin and Gera, Ariel and Boni, Odellia and Kantor, Yoav and Bloch, Gal and Levy, Omri and Abraham, Hadas and Barzilay, Nitzan and Shnarch, Eyal and Factor, Michael E. and Ofek-Koifman, Shila and Ta-Shma, Paula and Toledo, Assaf},
  eprint={2505.03452},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2025},
  url={https://arxiv.org/abs/2505.03452},
}
```

## License

The data is release under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

If you would like to see the detailed LICENSE click [here](LICENSE).

## Authors


- Author: Matan Orbach <matano@il.ibm.com>

[issues]: https://github.com/IBM/rag-hpo-bench/issues/new
[rag_config_results]: https://github.com/IBM/rag-hpo-bench/blob/main/data/rag_configurations_results.csv
