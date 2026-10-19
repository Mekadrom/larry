import pytest

from tests.helpers.duckdb_checks import env_connection, zero_rows_test


@pytest.fixture(scope="session")
def con():
    c, root = env_connection("SCOTUS_UNALIGNED")
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
    "exceptions_intact": """
                         WITH expected(docket, line_prefix) AS (VALUES ('19-292',
                                                                        'MS. TAIBLESON: Mr. Chief -- Mr. Chief Justice --'),
                                                                       ('19-292',
                                                                        'MS. CORKRAN: Oh, thank you, Your Honor.'),
                                                                       ('19-465', 'MR. LESSIG: Yes, I would.'),
                                                                       ('22-721',
                                                                        'MR. SIMPSON: I''m ready to jump in.'))
                         SELECT e.docket, e.line_prefix
                         FROM expected e
                         WHERE NOT EXISTS (SELECT 1
                                           FROM lines l
                                           WHERE l.docket = e.docket
                                             AND starts_with(l.line, e.line_prefix))""",
}


test_zero_rows = zero_rows_test(ZERO_ROWS)


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
