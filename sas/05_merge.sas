/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 6 — Combine datasets: PROC SQL joins + aggregations
 *
 * Mirrors Python: merge_flights_airports(), delay_by_carrier(),
 *                 delay_by_month(), airport_summary(), delay_breakdown_by_carrier()
 *
 * Input : WORK.FLIGHTS, WORK.AIRPORTS_RAW
 * Output: WORK.FLIGHTS_GEO, WORK.CARRIER_SUMMARY, WORK.MONTHLY_SUMMARY,
 *         WORK.AIRPORT_SUMMARY, WORK.CAUSE_SUMMARY, WORK.ROUTE_SUMMARY
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── 6a. Inner join flights with airport geo data ────────────────────────── */
PROC SQL;
    CREATE TABLE WORK.FLIGHTS_GEO AS
    SELECT
        f.*,
        a.Name                            AS airport_name,
        a.City                            AS origin_city,
        a.Country                         AS origin_country,
        a.Latitude                        AS origin_lat,
        a.Longitude                       AS origin_lon
    FROM  WORK.FLIGHTS      AS f
    INNER JOIN WORK.AIRPORTS_RAW AS a
        ON UPCASE(STRIP(f.origin)) = UPCASE(STRIP(a.IATA));
QUIT;

TITLE "Facility 6a — Flights joined with airport data (first 5 rows)";
PROC PRINT DATA=WORK.FLIGHTS_GEO (OBS=5)
    VAR origin airport_name origin_city origin_lat origin_lon dep_delay;
RUN;
TITLE;

/* ── 6b. Carrier-level aggregation — mirrors delay_by_carrier() ─────────── */
PROC SQL;
    CREATE TABLE WORK.CARRIER_SUMMARY AS
    SELECT
        op_unique_carrier                     AS carrier    LENGTH=2,
        COUNT(*)                              AS flight_count,
        MEAN(arr_delay)                       AS avg_arr_delay,
        MEAN(dep_delay)                       AS avg_dep_delay,
        MEAN(cancelled)  * 100                AS cancellation_pct,
        MEAN(carrier_delay)                   AS avg_carrier_delay,
        MEAN(weather_delay)                   AS avg_weather_delay,
        MEAN(nas_delay)                       AS avg_nas_delay,
        MEAN(security_delay)                  AS avg_security_delay,
        MEAN(late_aircraft_delay)             AS avg_late_aircraft_delay,
        MEAN(distance)                        AS avg_distance
    FROM  WORK.FLIGHTS
    GROUP BY op_unique_carrier
    ORDER BY avg_arr_delay DESC;
QUIT;

TITLE "Facility 6b — Carrier summary (all carriers, sorted by avg arrival delay)";
PROC PRINT DATA=WORK.CARRIER_SUMMARY NOOBS;
    FORMAT carrier           $CARRIER_FMT.
           flight_count      COMMA8.
           avg_arr_delay
           avg_dep_delay     8.2
           cancellation_pct  6.2
           avg_carrier_delay
           avg_weather_delay
           avg_nas_delay     8.2;
RUN;
TITLE;

/* ── 6c. Monthly aggregation — mirrors delay_by_month() ─────────────────── */
PROC SQL;
    CREATE TABLE WORK.MONTHLY_SUMMARY AS
    SELECT
        month,
        COUNT(*)              AS flight_count,
        MEAN(dep_delay)       AS avg_dep_delay,
        MEAN(arr_delay)       AS avg_arr_delay,
        MEAN(cancelled) * 100 AS cancellation_pct
    FROM  WORK.FLIGHTS
    GROUP BY month
    ORDER BY month;
QUIT;

TITLE "Facility 6c — Monthly delay trend";
PROC PRINT DATA=WORK.MONTHLY_SUMMARY NOOBS;
    FORMAT month         MONTH_FMT.
           flight_count  COMMA8.
           avg_dep_delay
           avg_arr_delay 8.2
           cancellation_pct 6.2;
RUN;
TITLE;

/* ── 6d. Airport-level summary with geo coordinates ─────────────────────── */
PROC SQL;
    CREATE TABLE WORK.AIRPORT_SUMMARY AS
    SELECT
        f.origin,
        a.Name                        AS airport_name,
        a.City                        AS city,
        a.Latitude                    AS lat,
        a.Longitude                   AS lon,
        COUNT(*)                      AS flight_count,
        MEAN(f.arr_delay)             AS avg_arr_delay,
        MEAN(f.dep_delay)             AS avg_dep_delay,
        MEAN(f.cancelled)   * 100     AS cancellation_pct
    FROM  WORK.FLIGHTS       AS f
    INNER JOIN WORK.AIRPORTS_RAW AS a
        ON UPCASE(STRIP(f.origin)) = UPCASE(STRIP(a.IATA))
    GROUP BY f.origin, a.Name, a.City, a.Latitude, a.Longitude
    ORDER BY flight_count DESC;
QUIT;

TITLE "Facility 6d — Top 20 busiest airports";
PROC PRINT DATA=WORK.AIRPORT_SUMMARY (OBS=20) NOOBS;
    FORMAT flight_count COMMA8.
           avg_arr_delay avg_dep_delay 8.2
           cancellation_pct 6.2;
RUN;
TITLE;

/* ── 6e. Delay cause breakdown per carrier — mirrors delay_breakdown_by_carrier() */
PROC SQL;
    CREATE TABLE WORK.CAUSE_SUMMARY AS
    SELECT
        op_unique_carrier             AS carrier           LENGTH=2,
        MEAN(carrier_delay)           AS avg_carrier,
        MEAN(weather_delay)           AS avg_weather,
        MEAN(nas_delay)               AS avg_nas,
        MEAN(security_delay)          AS avg_security,
        MEAN(late_aircraft_delay)     AS avg_late_aircraft,
        SUM(carrier_delay)
            + SUM(weather_delay)
            + SUM(nas_delay)
            + SUM(late_aircraft_delay) AS total_delay_minutes
    FROM  WORK.FLIGHTS
    GROUP BY op_unique_carrier
    ORDER BY avg_carrier DESC;
QUIT;

/* ── 6f. Route-level statistics (origin → dest pairs) ───────────────────── */
PROC SQL OUTOBS=30;
    CREATE TABLE WORK.ROUTE_SUMMARY AS
    SELECT
        CATX('-', origin, dest)   AS route        LENGTH=8,
        origin,
        dest,
        COUNT(*)                  AS flight_count,
        MEAN(arr_delay)           AS avg_arr_delay,
        MEAN(dep_delay)           AS avg_dep_delay,
        MEAN(cancelled) * 100     AS cancellation_pct
    FROM  WORK.FLIGHTS
    GROUP BY origin, dest
    HAVING COUNT(*) >= 50          /* routes with enough observations */
    ORDER BY avg_arr_delay DESC;
QUIT;

TITLE "Facility 6f — Top 30 routes by average arrival delay (min 50 flights)";
PROC PRINT DATA=WORK.ROUTE_SUMMARY NOOBS;
    FORMAT flight_count COMMA6. avg_arr_delay avg_dep_delay 8.2 cancellation_pct 6.2;
RUN;
TITLE;
