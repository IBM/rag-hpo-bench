# Missing Imports in rag-hpo-bench HPO Package

The following imports from the RAGEval package are required dependencies for the HPO package to function:

## Core Dependencies

### From `rageval.flows`
- `rageval.flows.dataset_id.DatasetID` - Used in: hpo_experiment.py, rag_runner.py, single_stage_tuner.py, tune_and_test_runner.py

### From `rageval.pipeline.api.data_model`
- `rageval.pipeline.api.data_model.EvaluationParams` - Used in: hpo_experiment.py
- `rageval.pipeline.api.data_model.ModelProvider` - Used in: search_space_factory.py


### From `rageval.experiment_setup`
- `rageval.experiment_setup.ExperimentSetup` - Used in: search_space.py

### From `rageval.utils`
- `rageval.utils.hash_utils.get_hash_dict` - Used in: search_space.py

## Summary

To use this package independently, you would need to either:
- Install RAGEval as a dependency (recommended)
- Extract and copy the required RAGEval modules listed above
- Create stub/mock implementations of the required interfaces