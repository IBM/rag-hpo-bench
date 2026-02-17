from pydantic import BaseModel, model_validator


class DataSamplingParams(BaseModel):
    """
    Limits the benchmark to a specific number of questions.
    Returns all questions if None (the default).
    """

    question_limit: int | None = None
    """
    Limits the documents to be the concatenation of the relevant documents
    and N (determined by document_factor) times non-relevant documents.
    """
    document_factor: int | None = None
    """
    An optional seed for reproducibility. Default is 43.
    """
    seed: int | None = None

    @model_validator(mode="after")
    def set_default_seed(self):
        """Ensure seed defaults to 43 if not provided."""
        if self.seed is None:
            self.seed = 43
        return self

    def as_id(self):
        result = ""
        if self.question_limit:
            result += f"q-{self.question_limit}"
        if self.document_factor:
            if result:
                result += "_"
            result += f"docs-factor-{self.document_factor}"
        if result:
            # sampling occurred, add the seed:
            result += f"_seed-{self.seed}"
        return result
