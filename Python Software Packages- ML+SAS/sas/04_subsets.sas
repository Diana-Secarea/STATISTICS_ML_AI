/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 4 — Data subsets (WHERE clause)
 * Creates focused analytical subsets that mirror Python filtered DataFrames.
 * Input : WORK.FLIGHTS
 * Output: WORK.SEVERE_DELAYS, WORK.CANCELLED_FLIGHTS, WORK.SUMMER_FLIGHTS,
 *         WORK.WORST_CARRIERS, WORK.BEST_CARRIERS, WORK.LONG_HAUL
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Severely delayed operated flights — prime candidates for rebooking algorithms */
DATA WORK.SEVERE_DELAYS;
    SET WORK.FLIGHTS;
    WHERE dep_delay >= 60 AND cancelled = 0;
RUN;

/* Cancelled flights only — used for cancellation model analysis */
DATA WORK.CANCELLED_FLIGHTS;
    SET WORK.FLIGHTS;
    WHERE cancelled = 1;
RUN;

/* Summer peak season — Python finding: Jul/Aug avg dep_delay > 18 min */
DATA WORK.SUMMER_FLIGHTS;
    SET WORK.FLIGHTS;
    WHERE month IN (7, 8);
RUN;

/* Worst-performing carriers by avg arrival delay (Python analysis result) */
DATA WORK.WORST_CARRIERS;
    SET WORK.FLIGHTS;
    WHERE op_unique_carrier IN ('F9', 'AA', 'B6');
RUN;

/* Best-performing carriers — for comparative benchmarking */
DATA WORK.BEST_CARRIERS;
    SET WORK.FLIGHTS;
    WHERE op_unique_carrier IN ('AS', 'HA');
RUN;

/* Long-haul flights (distance > 1500 miles) — different delay profile */
DATA WORK.LONG_HAUL;
    SET WORK.FLIGHTS;
    WHERE distance > 1500 AND cancelled = 0;
RUN;

/* Summary of all subset sizes — one SQL query for clean output */
TITLE "Facility 4 — Subset sizes";
PROC SQL;
    SELECT 'All cleaned flights'          AS subset LENGTH=35,
           COUNT(*)                       AS n      FORMAT=COMMA10.
    FROM WORK.FLIGHTS
    UNION ALL
    SELECT 'Severe delays (dep >= 60 min)', COUNT(*) FROM WORK.SEVERE_DELAYS
    UNION ALL
    SELECT 'Cancelled flights',             COUNT(*) FROM WORK.CANCELLED_FLIGHTS
    UNION ALL
    SELECT 'Summer (July + August)',         COUNT(*) FROM WORK.SUMMER_FLIGHTS
    UNION ALL
    SELECT 'Worst carriers (F9, AA, B6)',   COUNT(*) FROM WORK.WORST_CARRIERS
    UNION ALL
    SELECT 'Best carriers (AS, HA)',        COUNT(*) FROM WORK.BEST_CARRIERS
    UNION ALL
    SELECT 'Long-haul (distance > 1500 mi)',COUNT(*) FROM WORK.LONG_HAUL;
QUIT;
TITLE;

/* Descriptive comparison: severe vs. normal delays */
TITLE "Severe delays — key metrics";
PROC MEANS DATA=WORK.SEVERE_DELAYS N MEAN MEDIAN MAX MAXDEC=1;
    VAR dep_delay arr_delay total_cause_delay distance;
RUN;
TITLE;
