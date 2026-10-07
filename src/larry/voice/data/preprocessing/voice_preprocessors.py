from datasets import Dataset, DatasetDict

from larry.common.data.preprocessing.preprocessors import Preprocessor


class SCOTUSSegmentingPreprocessor(Preprocessor):
    def validate(self) -> None:
        pass

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        pass