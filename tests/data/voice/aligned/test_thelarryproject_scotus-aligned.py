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
    "nulls": """SELECT * EXCLUDE (audio)
                FROM seg
                WHERE audio.bytes IS NULL
                   OR text_original IS NULL
                   OR speaker_id IS NULL
                   OR ctc_mismatch IS NULL""",
    "bad_times": """SELECT docket, start_s, end_s, duration_s
                    FROM seg
                    WHERE end_s <= start_s
                       OR duration_s <= 0
                       OR abs(duration_s - (end_s - start_s)) > 0.002""",
    "over_cap": "SELECT docket, start_s, duration_s FROM seg WHERE duration_s > 25.0 + 0.001",
    "duplicates": "SELECT docket, start_s, count(*) FROM seg GROUP BY docket, start_s HAVING count(*) > 1",
    # text_original
    "no_alphanumeric": "SELECT docket, start_s, text_original FROM seg WHERE NOT regexp_matches(text_original, '[A-Za-z0-9]')",
    "stage_leak": r"""SELECT docket, start_s, text_original
                      FROM seg
                      WHERE regexp_matches(text_original, '(?i)\((?:laughter|pause|nods|indicating|inaudible|crosstalk)')""",
    "label_leak": r"""SELECT docket, start_s, text_original
                      FROM seg
                      WHERE regexp_matches(text_original,
                                           '\b(?:CHIEF JUSTICE|JUSTICE|GENERAL|MRS?\.|MS\.) [A-Z]{3,}[A-Z''-]*:')""",
    # speakers
    "justice_outside_tenure": """
                              SELECT s.speaker_id, s.docket, s.date_argued
                              FROM seg s
                                       JOIN bench b ON s.speaker_id = 'scotus:' || b.name
                              WHERE s.date_argued < b.start_date
                                 OR s.date_argued > coalesce(b.end_date, DATE '9999-12-31')""",
    "unknown_justice_id": """
                          SELECT DISTINCT s.speaker_id
                          FROM seg s
                                   LEFT JOIN bench b ON s.speaker_id = 'scotus:' || b.name
                          WHERE s.speaker_id LIKE 'scotus:%JUSTICE %'
                            AND NOT s.speaker_id LIKE 'scotus:adv:%'
                            AND NOT regexp_matches(s.speaker_id, '^scotus:\\d')
                            AND b.name IS NULL""",
    "unresolved_speaker_id": """
                             WITH allowed(speaker_id)
                                      AS (VALUES ('scotus:unknown')
                                 )
                             SELECT s.speaker_id,
                                    s.docket,
                                    count(*)                    AS n_segments,
                                    round(sum(s.duration_s), 1) AS seconds
                             FROM seg s ANTI JOIN allowed a
                             ON a.speaker_id = s.speaker_id
                             WHERE s.speaker_id IS NOT NULL
                               AND NOT regexp_matches(s.speaker_id
                                 , '^scotus:(CHIEF|JUSTICE|adv:)')
                             GROUP BY s.speaker_id, s.docket
                             ORDER BY n_segments DESC;
                             """,
    "wrong_attributions": """
                          WITH adv AS (SELECT docket,
                                              start_s,
                                              speaker_id,
                                              text_original,
                                              lag(speaker_id) OVER w AS prev_adv, lead(speaker_id) OVER w AS next_adv
                                       FROM seg
                                       WHERE speaker_id LIKE 'scotus:adv:%'
                                       WINDOW w AS (PARTITION BY docket ORDER BY start_s)),
                               allowed(docket, text_prefix) AS (VALUES ('19-292', 'Mr. Chief -- Mr. Chief Justice --'),
                                                                       ('19-465', 'Yes, I would.'))
                          SELECT a.docket,
                                 a.start_s,
                                 a.speaker_id,
                                 a.prev_adv AS surrounding,
                                 a.text_original
                          FROM adv a
                          WHERE a.prev_adv = a.next_adv
                            AND a.speaker_id != a.prev_adv
  AND NOT EXISTS (
      SELECT 1
      FROM allowed w
      WHERE w.docket = a.docket
        AND starts_with(a.text_original, w.text_prefix)
  )
                          ORDER BY a.docket, a.start_s;
                          """,
    # timing
    "time_travel": """
                   SELECT docket, start_s, prev_end
                   FROM (SELECT docket, start_s, lag(end_s) OVER (PARTITION BY docket ORDER BY start_s) AS prev_end
                         FROM seg)
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
                                                       count_if(duration_s < 0.3),
                                                       count_if(duration_s > 3 AND len(string_split(text_original, ' ')) / duration_s < 1.0),
                                                       count_if(duration_s > 1 AND
                                                                len(regexp_extract_all(text_original, '[A-Za-z0-9'']+')) /
                                                                duration_s > 8.0)
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
                       SELECT DISTINCT s.speaker_id
                       FROM seg s,
                            bench b
                       WHERE s.speaker_id LIKE 'scotus:adv:%'
                         AND s.speaker_id LIKE '% ' || split_part(b.name, ' ', -1)""").fetchall()
    new = {r[0] for r in rows} - KNOWN_ADV_JUSTICE_SURNAME
    assert not new, sorted(new)


def test_audio_decodes(con):
    rows = con.execute("SELECT audio.bytes, duration_s FROM seg USING SAMPLE 200 ROWS").fetchall()
    for data, secs in rows:
        dec = AudioDecoder(data)
        samples = dec.get_all_samples()
        assert samples.sample_rate == 16000
        assert abs(samples.data.shape[-1] / 16000 - secs) < 0.01
