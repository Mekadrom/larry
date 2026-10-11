import re
from typing import Any

from datasets import Dataset, DatasetDict
from datasets.features.features import Value, Features

from larry.common.data.preprocessing.mappers import SingleColumnMapper
from larry.common.data.utils import features_of
from larry.text.config.preprocessing.text_mapper_configs import SCOTUSTranscriptPdfTextMapperConfig
from larry.voice.config.preprocessing.scotus.scotus_mapper_configs import SCOTUSSpeechSegmentationAligningMapperConfig
from larry.voice.config.preprocessing.voice_configs import SCOTUSTranscriptConfig
from larry.voice.data.preprocessing import scotus_transcript as scotus_transcript
from larry.voice.data.preprocessing.normalization import TextNormalizer, SCOTUSTextNormalizer
from larry.voice.data.preprocessing.scotus_transcript import SCOTUSTranscript
from larry.voice.data.preprocessing.transcript import pdf_bytes_to_text, Transcript, Turn
from larry.voice.data.preprocessing.voice_mappers import SpeechSegmentationAligningMapper

# reporter typos in speaker labels, checked against the surrounding dialog
_LABEL_FIXES: dict[tuple[str, str], str] = {
    ("11-204", "CHIEF JUSTICE BREYER"): "CHIEF JUSTICE ROBERTS",  # "Mr. Goldstein, 3 minutes." is a time call
    ("16-1275", "CHIEF JUSTICE GORSUCH"): "CHIEF JUSTICE ROBERTS",  # resumes Roberts's "we've got too many --"
    ("22-451", "GENERAL GORSUCH"): "JUSTICE GORSUCH",
    ("16-299", "MS. KAGAN"): "JUSTICE KAGAN",
    ("22-1219", "MR. KAGAN"): "JUSTICE KAGAN",
    ("17-988", "Mr. SOTOMAYOR"): "JUSTICE SOTOMAYOR",
    ("22-800", "JUSTICE PRELOGAR"): "GENERAL PRELOGAR",
    ("11-551", "JUSTICE PHILLIPS"): "JUSTICE KENNEDY",
    ("23-583", "MR. DEGER SEN"): "MR. DEGER-SEN",
    ("15-5040", "MS. EISENSTEIN"): "MR. EISENBERG",
    ("24-416", "MS. ROSE"): "MS. ROSS",
    ("15-109", "MR. RAMIREZ"): "MR. MARTINEZ",
    ("23-909", "MR. REIGN"): "MR. FEIGIN",
    ("11-889", "MR. CLEMENT"): "MR. ROTHFELD",
    ("24-297", "MS. WILSON"): "MS. HARRIS",
    ("23-477", "MR. DAVIE"): "MR. RICE",
    ("12-930", "MR. MITCHELL"): "MR. FLEMING",
    ("15-927", "MS. SULLIVAN"): "MR. WAXMAN",
    ("12-416", "MR. KATZ"): "MR. WEINBERGER",
    ("14-844", "MR. KIMBERLY"): "MR. SHELLEY",
    ("14-844", "MR. SULLIVAN"): "MS. SAHARSKY",
    ("10-6", "MS. BLATT"): "MR. CRUZ",
    ("13-9972", "MR. KELLER"): "MR. O'CONNOR",
    ("10-879", "MS. SMITH"): "MS. HARRINGTON",
}
_TITLE_FIX = re.compile(r"^(M\s?RS?|MS)(?:\s*\.\s*|\s+)(?=\S)", re.I)
_LABEL_FIXES_ANY: dict[str, str] = {
    "CHIEF JUSTICE ROBERT": "CHIEF JUSTICE ROBERTS",
    "CHIEF JUDGE ROBERTS": "CHIEF JUSTICE ROBERTS",
    "MR. BREYER": "JUSTICE BREYER",
}


class SCOTUSTranscriptPdfTextMapper(SingleColumnMapper[bytes, str, SCOTUSTranscriptPdfTextMapperConfig]):
    def __init__(self, provenance_columns, config):
        super().__init__(provenance_columns, config)
        self.remove_columns = [c for c in self.remove_columns if c != self.config.input_column]

    def validate(self) -> None:
        if not self.config.input_column:
            raise ValueError(f"input_column must be specified for {self.__class__.__name__}")
        if not self.config.output_column:
            raise ValueError(f"output_column must be specified for {self.__class__.__name__}")

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        transcripts = []
        reattributions = []
        for pdf_bytes, docket, date_argued in zip(
                batch[self.config.input_column],
                batch[self.config.docket_column],
                batch[self.config.date_argued_column]
        ):
            tr = scotus_transcript.SCOTUSTranscript(SCOTUSTranscriptConfig(), docket, date_argued)
            tr.parse_from_raw(pdf_bytes_to_text(pdf_bytes))
            reattributions.append(tr.reattributions)

            speaker_normalized_turns = []
            for turn in tr.turns:
                speaker = _LABEL_FIXES.get((docket, turn.speaker), turn.speaker)
                speaker = _TITLE_FIX.sub(lambda m: re.sub(r"\s", "", m.group(1)).upper() + ". ", speaker)
                speaker = _LABEL_FIXES_ANY.get(speaker, speaker)
                speaker_normalized_turns.append(f"{speaker}: {turn.text}")

            transcripts.append("\n".join(speaker_normalized_turns))
        return {
            self.config.output_column: transcripts,
            self.config.reattributions_column: [
                [
                    {"before": before, "after": after, "text": text}
                    for before, after, text in reattrs
                ]
                for reattrs in reattributions
            ],
        }

    def preprocessing_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        features = super().preprocessing_features(dataset)
        if features is None:
            features = features_of(dataset).copy()
        for column in self.remove_columns:
            features.pop(column, None)
        features.update({
            self.config.output_column: Value("string"),
            self.config.reattributions_column: [{
                "before": Value("string"),
                "after": Value("string"),
                "text": Value("string"),
            }]
        })
        return features


class SCOTUSSpeechSegmentationAligningMapper(
    SpeechSegmentationAligningMapper[SCOTUSSpeechSegmentationAligningMapperConfig]
):
    def make_normalizer(self) -> TextNormalizer:
        return SCOTUSTextNormalizer()

    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        docket = batch[self.config.docket_column][index]
        date_argued = batch[self.config.date_argued_column][index]
        tr = SCOTUSTranscript(SCOTUSTranscriptConfig(), docket, date_argued)
        tr.parse_from_turns(turns)
        tr.raw_text = pdf_bytes_to_text(batch[self.config.pdf_bytes_column][index])
        return tr
