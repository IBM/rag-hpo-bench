from rageval.pipeline.api.data_model import ModelProvider

from rag_hpo_bench.hpo.search_space import (
    RagParameterName,
    SearchSpace,
    SearchSpaceParameter,
)

_DATA_PIPELINE = "data_pipeline"
_INFERENCE_PIPELINE = "inference_pipeline"

_INDEXING = "indexing"
_RETRIEVAL = "retrieval"
_GENERATION = "generation"


def new_search_space(
    chunk_sizes: int | list[int],
    embedding_models: list[str] | list[ModelProvider],
    generative_models: list[str] | list[ModelProvider],
    chunk_overlaps: list[int],
):
    vector_space_configs = [
        {RagParameterName.VENDOR: "milvus", RagParameterName.EMBEDDING_MODEL: model}
        for model in embedding_models
    ]

    parameters = [
        SearchSpaceParameter(
            path=[_DATA_PIPELINE, "type"],
            values="BasicDataPipeline",
        ),
        SearchSpaceParameter(
            path=[
                _DATA_PIPELINE,
                "params",
                _INDEXING,
                RagParameterName.VECTOR_SPACE,
            ],
            values=vector_space_configs,
        ),
        SearchSpaceParameter(
            path=[
                _DATA_PIPELINE,
                "params",
                _INDEXING,
                RagParameterName.CHUNK_SIZE,
            ],
            values=chunk_sizes,
        ),
        SearchSpaceParameter(
            path=[
                _DATA_PIPELINE,
                "params",
                _INDEXING,
                RagParameterName.CHUNK_OVERLAP,
            ],
            values=chunk_overlaps,
        ),
        SearchSpaceParameter(
            path=[
                _DATA_PIPELINE,
                "params",
                _INDEXING,
                RagParameterName.CHUNK_UNIT,
            ],
            values="character",
        ),
        SearchSpaceParameter(
            path=[_INFERENCE_PIPELINE, "type"],
            values="BasicInferencePipeline",
        ),
        SearchSpaceParameter(
            path=[
                _INFERENCE_PIPELINE,
                "params",
                _RETRIEVAL,
                RagParameterName.TOP_K,
            ],
            values=10,
        ),
        SearchSpaceParameter(
            path=[
                _INFERENCE_PIPELINE,
                "params",
                _GENERATION,
                RagParameterName.TEMPERATURE,
            ],
            values=0,
        ),
        SearchSpaceParameter(
            path=[
                _INFERENCE_PIPELINE,
                "params",
                _GENERATION,
                RagParameterName.MIN_NEW_TOKENS,
            ],
            values=1,
        ),
        SearchSpaceParameter(
            path=[
                _INFERENCE_PIPELINE,
                "params",
                _GENERATION,
                RagParameterName.MAX_NEW_TOKENS,
            ],
            values=500,
        ),
        SearchSpaceParameter(
            path=[
                _INFERENCE_PIPELINE,
                "params",
                _GENERATION,
                RagParameterName.GENERATIVE_MODEL,
            ],
            values=generative_models,
        ),
    ]
    return SearchSpace(parameters=parameters)


def create_paper_search_space():
    """
    TODO: update to paper search space
    """
    return new_search_space(
        chunk_sizes=512,
        chunk_overlaps=[128, 256],
        embedding_models=["local/e5_large"],
        generative_models=[
            "azure/gpt_4o",
            "local/llama_3_1_8b_instruct",
            "local/granite_3_1_8b_instruct",
        ],
    )
