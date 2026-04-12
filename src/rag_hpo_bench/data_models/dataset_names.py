from enum import StrEnum, unique


@unique
class DatasetName(StrEnum):
    """Dataset names used in RAG HPO benchmarking."""

    AIArxiv = "AIArxiv"
    BioASQ = "BioASQ"
    ClapNQ = "ClapNQ"
    MiniWiki = "MiniWiki"
    WatsonxQA = "WatsonxQA"
