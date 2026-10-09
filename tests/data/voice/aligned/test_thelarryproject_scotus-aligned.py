import os

import duckdb
import pytest
from torchcodec.decoders import AudioDecoder

from larry.voice.data.preprocessing import scotus_transcript


@pytest.fixture(scope="session")
def con():
    root = os.environ.get("SCOTUS_ALIGNED")
    if root is None:
        pytest.skip("SCOTUS_ALIGNED not set")
    c = duckdb.connect()
    c.execute(f"CREATE VIEW seg AS SELECT * FROM read_parquet('{root}/**/*.parquet')")
    c.execute("CREATE TABLE bench (name VARCHAR, start_date DATE, end_date DATE)")
    c.executemany("INSERT INTO bench VALUES (?, ?, ?)", scotus_transcript._BENCH)
    return c

ZERO_ROWS = {
    # structure
    "nulls": """SELECT * EXCLUDE (audio) FROM seg
                WHERE audio.bytes IS NULL OR text IS NULL OR speaker_id IS NULL OR avg_word_score IS NULL""",
    "bad_times": """SELECT docket, start_s, end_s, audio_seconds FROM seg
                    WHERE end_s <= start_s OR audio_seconds <= 0
                       OR abs(audio_seconds - (end_s - start_s)) > 0.002""",
    "over_cap": "SELECT docket, start_s, audio_seconds FROM seg WHERE audio_seconds > 25.0 + 0.001",
    "duplicates": "SELECT docket, start_s, count(*) FROM seg GROUP BY docket, start_s HAVING count(*) > 1",
    # text
    "no_alphanumeric": "SELECT docket, start_s, text FROM seg WHERE NOT regexp_matches(text, '[A-Za-z0-9]')",
    "stage_leak": r"""SELECT docket, start_s, text FROM seg
                      WHERE regexp_matches(text, '(?i)\((?:laughter|pause|nods|indicating|inaudible|crosstalk)')""",
    "label_leak": r"""SELECT docket, start_s, text FROM seg
                      WHERE regexp_matches(text, '\b(?:CHIEF JUSTICE|JUSTICE|GENERAL|MRS?\.|MS\.) [A-Z]{3,}[A-Z''-]*:')""",
    # speakers
    "justice_outside_tenure": """
        SELECT s.speaker_id, s.docket, s.date_argued FROM seg s
        JOIN bench b ON s.speaker_id = 'scotus:' || b.name
        WHERE s.date_argued < b.start_date OR s.date_argued > coalesce(b.end_date, DATE '9999-12-31')""",
    "unknown_justice_id": """
        SELECT DISTINCT s.speaker_id FROM seg s
        LEFT JOIN bench b ON s.speaker_id = 'scotus:' || b.name
        WHERE s.speaker_id LIKE 'scotus:%JUSTICE %' AND NOT s.speaker_id LIKE 'scotus:adv:%'
          AND NOT regexp_matches(s.speaker_id, '^scotus:\\d') AND b.name IS NULL""",
    # timing
    "time_travel": """
        SELECT docket, start_s, prev_end FROM (
          SELECT docket, start_s, lag(end_s) OVER (PARTITION BY docket ORDER BY start_s) AS prev_end FROM seg)
        WHERE prev_end - start_s > 0.25""",
}

@pytest.mark.parametrize("sql", ZERO_ROWS.values(), ids=ZERO_ROWS.keys())
def test_zero_rows(con, sql):
    rows = con.execute(sql).fetchall()
    assert not rows, f"{len(rows)} rows, e.g. {rows[:10]}"

def test_rates(con):
    n, fallback, tiny, slow, fast = con.execute(r"""
                                                SELECT count(*),
                                                       count_if(regexp_matches(speaker_id, '^scotus:\d')),
                                                       count_if(audio_seconds < 0.3),
                                                       count_if(audio_seconds > 3 AND len(string_split(text, ' ')) / audio_seconds < 1.0),
                                                       count_if(audio_seconds > 1 AND len(regexp_extract_all(text, '[A-Za-z0-9'']+')) / audio_seconds > 8.0)
                                                FROM seg""").fetchone()
    assert fallback / n < 0.001, fallback
    assert tiny / n < 0.01, tiny
    assert slow / n < 0.005, slow  # long silences left inside segments
    assert fast / n < 0.002, fast  # text squeezed into too little audio: misalignment

KNOWN_ADV_JUSTICE_SURNAME = {
    "scotus:adv:ELENA KAGAN",
    "scotus:adv:EUGENE SCALIA",
    "scotus:adv:VICKI JACKSON",
    "scotus:adv:SEAN KENNEDY",
    "scotus:adv:NICHOLAS KENNEDY",
    "scotus:adv:MATTHEW ROBERTS",
    "scotus:adv:JOHN ROBERTS",
}

def test_adv_with_justice_surname(con):
    rows = con.execute("""
        SELECT DISTINCT s.speaker_id FROM seg s, bench b
        WHERE s.speaker_id LIKE 'scotus:adv:%' AND s.speaker_id LIKE '% ' || split_part(b.name, ' ', -1)""").fetchall()
    new = {r[0] for r in rows} - KNOWN_ADV_JUSTICE_SURNAME
    assert not new, sorted(new)

def test_audio_decodes(con):
    rows = con.execute("SELECT audio.bytes, audio_seconds FROM seg USING SAMPLE 200 ROWS").fetchall()
    for data, secs in rows:
        dec = AudioDecoder(data)
        samples = dec.get_all_samples()
        assert samples.sample_rate == 16000
        assert abs(samples.data.shape[-1] / 16000 - secs) < 0.01
