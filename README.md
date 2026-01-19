<!-- This should be the location of the title of the repository, normally the short name -->
# RAG HPO Bench (Under construction)

<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->


This project contains code accompanying the paper
[An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation](https://arxiv.org/abs/2505.03452).

The paper release a dataset called [RAG-HPO-bench](https://huggingface.co/datasets/matanor/rag-hpo-bench) which includes RAG results of multiple RAG
configurations over a search space of 162 parameter combinations. 
Building on these results, further research can explore
new HPO techniques without incurring the substantial cost
of running many RAG configuration across datasets.

This repository contains code examples with analysis using the RAG-HPO-bench dataset.

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

## License

The data is release under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

If you would like to see the detailed LICENSE click [here](LICENSE).

## Authors


- Author: Matan Orbach <matano@il.ibm.com>

[issues]: https://github.com/IBM/rag-hpo-bench/issues/new
[rag_config_results]: https://github.com/IBM/rag-hpo-bench/blob/main/data/rag_configurations_results.csv
