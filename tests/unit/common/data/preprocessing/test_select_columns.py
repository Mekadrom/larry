from datasets import Dataset, DatasetDict

from larry.common.config.data.preprocessing.preprocessor_configs import SelectColumnsPreprocessorConfig
from larry.common.data.preprocessing.preprocessors import SelectColumnsPreprocessor
from tests.helpers.common import provenance


def test_every_split_keeps_only_row_and_id_by_default() -> None:
    split = Dataset.from_dict({"text": ["x"], "id": ["a"], "duration_s": [1.0], "row": [0]})
    dataset = DatasetDict({"train": split, "validation": split})

    out = SelectColumnsPreprocessor(provenance(), SelectColumnsPreprocessorConfig()).preprocess_dataset(dataset)

    assert {name: s.column_names for name, s in out.items()} == {"train": ["row", "id"], "validation": ["row", "id"]}
