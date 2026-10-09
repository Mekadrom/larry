import os

import duckdb
import pytest


@pytest.fixture(scope="session")
def con():
    root = os.environ.get("SCOTUS_UNALIGNED")
    if root is None:
        pytest.skip("SCOTUS_UNALIGNED not set")
    c = duckdb.connect()
    c.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{root}/**/*.parquet')")
    c.execute("""
              CREATE VIEW lines AS
              SELECT term, docket, date_argued, l AS line, row_number() OVER (PARTITION BY docket) AS line_no
              FROM t,
                   unnest(string_split(transcript, chr(10))) AS u(l)
              """)
    return c


ZERO_ROWS = {
    "merged_dash_labels": r"""
                          SELECT docket
                          FROM t
                          WHERE regexp_matches(transcript,
                                               '\b(?:CHIEF JUSTICE|JUSTICE|GENERAL|MRS?\.|MS\.) [A-Z][\p{L}''-]+ --')""",
    "non_roberts_chief": """
                         SELECT docket, line
                         FROM lines
                         WHERE regexp_matches(line, '^CHIEF JUSTICE [A-Z]+:')
                           AND NOT line LIKE 'CHIEF JUSTICE ROBERTS:%'""",
    "triple_dash": "SELECT docket, line FROM lines WHERE line LIKE '%---%'",
}


@pytest.mark.parametrize("sql", ZERO_ROWS.values(), ids=ZERO_ROWS.keys())
def test_zero_rows(con, sql):
    rows = con.execute(sql).fetchall()
    assert not rows, f"{len(rows)} rows, e.g. {rows[:10]}"


def test_words_per_second(con):
    lo, hi = con.execute("""
                         SELECT min(len(string_split(transcript, ' ')) / duration_s),
                                max(len(string_split(transcript, ' ')) / duration_s)
                         FROM t""").fetchone()
    assert 1.8 < lo and hi < 4.0, (lo, hi)


KNOWN_RARE = {
    "CHIEF JUSTICE", "GENERAL KAGAN", "GETCHELL", "GINSBURG", "GUPTA",
    "JSUTICE BREYER", "JUDGE KAVANAUGH", "JUDGE KENNEDY", "JUDGE SOTOMAYOR",
    "JUST ALITO", "JUST GORSUCH", "JUST KAGAN", "JUST SOTOMAYOR",
    "JUSTICE BARRET", "JUSTICE BEYER", "JUSTICE BRYER", "JUSTICE GORUSCH", "JUSTICE KENNEY",
    "JUSTICE SCLIA", "JUSTICE SOTOYMAYOR", "JUSTICE TO KAVANAUGH", "JUSTICIE SOTOMAYOR",
    "MR. BARBERT", "MR. BELLENGER", "MR. BLEMENT", "MR. BROWN", "MR. CHELSER", "MR. CHEMERINKSY",
    "MR. DEGER SEN", "MR. DESANCTIS", "MR. ELLISS", "MR. GUARNERI", "MR. GUARNIER",
    "MR. HALLWARD-DREIMEIER", "MR. HARRINGTON", "MR. HEYTEN", "MR. HICK", "MR. MCALLISTER",
    "MR. MCGINLEY", "MR. METLISTSKY", "MR. MOOPAN", "MR. O'CONNELL", "MR. OlSON", "MR. PHILIPS",
    "MR. PRELOGAR", "MR. RAMIREZ", "MR. REIGN", "MR. RUSSEL", "MR. SACHS", "MR. SELLER",
    "MR. SKATSKEE", "MR. SYNDER", "MR. THEIRMAN", "MR. TSYETLIN", "MR. UNIKOWKSY",
    "MS. BROWNING", "MS. GOLDBERG", "MS. JACOB", "MS. KOMP", "MS. PARKER", "MS. RAVES",
    "MS. ROSE", "MS. SANDERS", "MS. SCODRO", "MS. WILSON",
}


def test_rare_labels(con):
    rows = con.execute("""
                       SELECT split_part(line, ':', 1) AS label
                       FROM lines
                       GROUP BY label
                       HAVING count(*) <= 2""").fetchall()
    new = {r[0] for r in rows} - KNOWN_RARE
    assert not new, sorted(new)
