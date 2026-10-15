from typing import Any

from datasets import Dataset, DatasetDict
from datasets.features.features import Value, Features

from larry.common.data.preprocessing.mappers import Mapper
from larry.text.config.preprocessing.text_mapper_configs import SCOTUSTranscriptPdfTextMapperConfig
from larry.voice.config.preprocessing.scotus.scotus_mapper_configs import SCOTUSSpeechSegmentationAligningMapperConfig
from larry.voice.config.preprocessing.voice_configs import SCOTUSTranscriptConfig
from larry.voice.data.preprocessing import scotus_transcript as scotus_transcript
from larry.voice.data.preprocessing.normalization import TextNormalizer, SCOTUSTextNormalizer
from larry.voice.data.preprocessing.scotus_transcript import SCOTUSTranscript
from larry.voice.data.preprocessing.transcript import pdf_bytes_to_text, Transcript, Turn
from larry.voice.data.preprocessing.voice_mappers import SpeechSegmentationAligningMapper


class SCOTUSTranscriptPdfTextMapper(Mapper[SCOTUSTranscriptPdfTextMapperConfig]):
    """Specialized instance of ``Mapper`` for producing semi-normalized SCOTUS transcript text.

    Takes a PDF (in bytes) and converts it to stage-direction/play script-style speaker-turn ordered dialog with
    directions, headers, and the ending word index removed.

    E.g., the PDF transcript when copied will look something like this (after the appearances and docket abstract):\
    ```
    1                     P R O C E E D I N G S
    2                                      (10:03 a.m.)
    3               CHIEF JUSTICE ROBERTS: We will hear
    4    argument this morning in Case 25-498, Anderson
    5    versus Intel Corporation Investment Policy
    6    Committee.
    7        Mr. Wessler.
    8        ORAL ARGUMENT OF MATTHEW W.H. WESSLER
    9            ON BEHALF OF THE PETITIONERS
    10       MR. WESSLER: Mr. Chief Justice, and
    11   may it please the Court:

    ...
    ```
    and normalizes it to something like this:
    ```
    CHIEF JUSTICE ROBERTS: We will hear argument this morning in Case 25-498, Anderson versus Intel Corporation \\
    Investment Policy Committee
    MR. WESSLER: Mr. Chief Justice, and may it please the Court:

    ...
    ```
    """

    def preprocess_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Performs the preprocessing logic required for this step on a batch of PDF transcripts."""
        transcripts = []
        reattributions = []

        pdf_bytes_in = batch[self.config.pdf_bytes_column]
        docket_in = batch[self.config.docket_column]
        date_argued_in = batch[self.config.date_argued_column]
        for pdf_bytes, docket, date_argued in zip(pdf_bytes_in, docket_in, date_argued_in):
            # delegates most of the real work to a somewhat generalized transcript parser that has been overridden
            # for SCOTUS-specific idiosyncrasies
            tr = scotus_transcript.SCOTUSTranscript(SCOTUSTranscriptConfig(), docket, date_argued)
            tr.parse_from_raw(pdf_bytes_to_text(pdf_bytes))
            transcripts.append("\n".join([f"{t.speaker}: {t.text}" for t in tr.turns]))
            reattributions.append(tr.reattributions)
        return {
            self.config.transcript_output_column: transcripts,
            self.config.reattributions_output_column: [
                [
                    {"before": before, "after": after, "text": text}
                    for before, after, text in reattrs
                ]
                for reattrs in reattributions
            ],
        }

    def output_features(self, dataset: Dataset | DatasetDict) -> Features | None:
        """Defines the output features.

        The ``transcript`` column is a simple string, and the ``reattributions`` column is a struct of three string
        fields, ``before``, ``after``, and ``text``.
        """
        return Features({
            self.config.transcript_output_column: Value("string"),
            self.config.reattributions_output_column: [{
                "before": Value("string"),
                "after": Value("string"),
                "text": Value("string"),
            }]
        })


class SCOTUSSpeechSegmentationAligningMapper(
    SpeechSegmentationAligningMapper[SCOTUSSpeechSegmentationAligningMapperConfig]
):
    """Configuring override of ``SpeechSegmentationAligningMapper``.

     Defines what instance of ``TextNormalizer`` and ``Transcript`` to use, providing instances of each to the
     superclass containing the real logic for splitting a transcript and its audio into corresponding aligned segments.
     """

    def make_normalizer(self) -> TextNormalizer:
        """Instantiates a simple override implementation of ``TextNormalizer``, ``SCOTUSTextNormalizer.

        ``SCOTUSTextNormalizer`` overrides label derivation for numbers specifically so statutes and dockets get their
        correct pronunciation elaborated.
        """
        return SCOTUSTextNormalizer()

    def new_transcript(self, turns: list[Turn], batch: dict[str, list[Any]], index: int) -> Transcript:
        """Instantiates the implementation of ``Transcript`` specific to SCOTUS preprocessing, ``SCOTUSTranscript.

        Also calls does some of the parsing logic, since the order of operations in regard to taking the raw transcript
        and producing lines or saving different fields on the ``Transcript`` instance varies between implementations.
        """
        docket = batch[self.config.docket_column][index]
        date_argued = batch[self.config.date_argued_column][index]

        tr = SCOTUSTranscript(SCOTUSTranscriptConfig(), docket, date_argued)
        tr.parse_from_turns(turns)
        tr.raw_text = pdf_bytes_to_text(batch[self.config.pdf_bytes_column][index])
        return tr
