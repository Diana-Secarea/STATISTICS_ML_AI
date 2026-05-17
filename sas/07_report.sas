/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 8 — Report procedures (PROC REPORT)
 * Produces formatted tabular outputs suitable for the Word document.
 *
 * Input : WORK.CARRIER_SUMMARY, WORK.MONTHLY_SUMMARY, WORK.FLIGHTS
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── 8a. Airline performance league table ────────────────────────────────── */
TITLE  "Facility 8a — US Airline Delay Performance: 2024 Summary";
TITLE2 "Sorted by average arrival delay (worst → best)";
PROC REPORT DATA=WORK.CARRIER_SUMMARY NOWD HEADLINE HEADSKIP;

    COLUMN carrier flight_count
           avg_arr_delay avg_dep_delay
           cancellation_pct
           avg_nas_delay avg_carrier_delay avg_weather_delay;

    DEFINE carrier              / DISPLAY "Airline"
                                  FORMAT=$CARRIER_FMT. WIDTH=22;
    DEFINE flight_count         / DISPLAY "Flights"
                                  FORMAT=COMMA8.       WIDTH=10;
    DEFINE avg_arr_delay        / DISPLAY "Avg Arr Delay (min)"
                                  FORMAT=8.1           WIDTH=20;
    DEFINE avg_dep_delay        / DISPLAY "Avg Dep Delay (min)"
                                  FORMAT=8.1           WIDTH=20;
    DEFINE cancellation_pct     / DISPLAY "Cancel %"
                                  FORMAT=6.2           WIDTH=10;
    DEFINE avg_nas_delay        / DISPLAY "NAS Delay (min)"
                                  FORMAT=8.1           WIDTH=16;
    DEFINE avg_carrier_delay    / DISPLAY "Carrier Delay (min)"
                                  FORMAT=8.1           WIDTH=18;
    DEFINE avg_weather_delay    / DISPLAY "Weather Delay (min)"
                                  FORMAT=8.1           WIDTH=18;

    /* Highlight worst carrier (highest avg_arr_delay — first row) */
    COMPUTE avg_arr_delay;
        IF avg_arr_delay > 10 THEN
            CALL DEFINE(_COL_, 'STYLE', 'STYLE=[BACKGROUND=LIGHTSALMON]');
    ENDCOMP;

RUN;
TITLE;

/* ── 8b. Monthly trend report with computed column ───────────────────────── */
TITLE  "Facility 8b — Monthly Delay Trend with Arrival–Departure Gap";
TITLE2 "A positive gap means flights recover time in the air; negative = lose time";
PROC REPORT DATA=WORK.MONTHLY_SUMMARY NOWD HEADLINE HEADSKIP;

    COLUMN month flight_count avg_dep_delay avg_arr_delay cancellation_pct delay_gap;

    DEFINE month            / DISPLAY "Month"
                              FORMAT=MONTH_FMT. WIDTH=6;
    DEFINE flight_count     / DISPLAY "Flights"
                              FORMAT=COMMA8.    WIDTH=10;
    DEFINE avg_dep_delay    / DISPLAY "Avg Dep Delay"
                              FORMAT=8.1        WIDTH=14;
    DEFINE avg_arr_delay    / DISPLAY "Avg Arr Delay"
                              FORMAT=8.1        WIDTH=14;
    DEFINE cancellation_pct / DISPLAY "Cancel %"
                              FORMAT=6.2        WIDTH=10;
    DEFINE delay_gap        / COMPUTED "Arr-Dep Gap"
                              FORMAT=8.1        WIDTH=12;

    COMPUTE delay_gap;
        delay_gap = avg_arr_delay - avg_dep_delay;
        /* flag months where flights arrive later than they departed */
        IF delay_gap > 2 THEN
            CALL DEFINE(_COL_, 'STYLE', 'STYLE=[FOREGROUND=DARKRED FONT_WEIGHT=BOLD]');
    ENDCOMP;

    /* Bold the summer peak rows */
    COMPUTE month;
        IF month IN (7, 8) THEN
            CALL DEFINE(_ROW_, 'STYLE', 'STYLE=[BACKGROUND=LIGHTYELLOW]');
    ENDCOMP;

RUN;
TITLE;

/* ── 8c. Primary delay cause cross-tab ──────────────────────────────────── */
TITLE  "Facility 8c — Delay Cause Frequency by Carrier";
PROC REPORT DATA=WORK.FLIGHTS (WHERE=(primary_cause NE 'None'))
    NOWD HEADLINE HEADSKIP;

    COLUMN op_unique_carrier primary_cause n pct;

    DEFINE op_unique_carrier / GROUP   "Airline"
                               FORMAT=$CARRIER_FMT. WIDTH=22;
    DEFINE primary_cause     / GROUP   "Primary Cause" WIDTH=14;
    DEFINE n                 / N       "Flights"
                               FORMAT=COMMA8.        WIDTH=10;
    DEFINE pct               / COMPUTED "% of Carrier" FORMAT=6.1 WIDTH=14;

    COMPUTE pct;
        /* percent within airline group */
        pct = (n / n.sum) * 100;
    ENDCOMP;

RUN;
TITLE;

/* ── 8d. Top 10 airports — geo-enriched summary ─────────────────────────── */
TITLE  "Facility 8d — Top 10 Origin Airports by Flight Volume";
PROC REPORT DATA=WORK.AIRPORT_SUMMARY (OBS=10) NOWD HEADLINE;

    COLUMN origin airport_name city flight_count avg_arr_delay cancellation_pct;

    DEFINE origin           / DISPLAY "IATA"        WIDTH=5;
    DEFINE airport_name     / DISPLAY "Airport"     WIDTH=35;
    DEFINE city             / DISPLAY "City"        WIDTH=20;
    DEFINE flight_count     / DISPLAY "Flights"
                              FORMAT=COMMA8.         WIDTH=10;
    DEFINE avg_arr_delay    / DISPLAY "Avg Arr Delay"
                              FORMAT=8.1             WIDTH=14;
    DEFINE cancellation_pct / DISPLAY "Cancel %"
                              FORMAT=6.2             WIDTH=10;

RUN;
TITLE;
