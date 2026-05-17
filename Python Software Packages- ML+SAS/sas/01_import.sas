/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 1 — Create SAS datasets from external files (PROC IMPORT)
 * Input : flight_data_2024.csv  (~7 M rows), Airports-Only.csv
 * Output: WORK.FLIGHTS_RAW, WORK.AIRPORTS_RAW, WORK.FLIGHTS_SAMPLE (100 k)
 * ══════════════════════════════════════════════════════════════════════════════ */

PROC IMPORT
    DATAFILE = "&DATA_DIR./flight_data_2024.csv"
    OUT      = WORK.FLIGHTS_RAW
    DBMS     = CSV
    REPLACE;
    GETNAMES    = YES;
    GUESSINGROWS = 3000;
RUN;

PROC IMPORT
    DATAFILE = "&DATA_DIR./Airports-Only.csv"
    OUT      = WORK.AIRPORTS_RAW
    DBMS     = CSV
    REPLACE;
    GETNAMES    = YES;
    GUESSINGROWS = 3000;
RUN;

/*
 * Draw a reproducible simple random sample of &SAMPLE_N flights.
 * SEED=&SEED mirrors Python: flights_full.sample(n=100_000, random_state=42)
 */
PROC SURVEYSELECT
    DATA   = WORK.FLIGHTS_RAW
    OUT    = WORK.FLIGHTS_SAMPLE
    METHOD = SRS
    N      = &SAMPLE_N
    SEED   = &SEED
    NOPRINT;
RUN;

/* Quick sanity check — shape and first rows */
TITLE "Facility 1 — Imported dataset dimensions";
PROC CONTENTS DATA=WORK.FLIGHTS_SAMPLE VARNUM SHORT; RUN;

TITLE "Facility 1 — First 5 rows of flight sample";
PROC PRINT DATA=WORK.FLIGHTS_SAMPLE (OBS=5); RUN;
TITLE;
