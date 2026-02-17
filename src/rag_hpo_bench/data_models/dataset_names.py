from enum import Enum, unique


@unique
class DatasetName(str, Enum):
    """Dataset names used in RAG HPO benchmarking."""
    
    AIArxiv = "AIArxiv"
    BioASQ = "BioASQ"
    ClapNQ = "ClapNQ"
    MiniWiki = "MiniWiki"
    WatsonxQA = "WatsonxQA"
