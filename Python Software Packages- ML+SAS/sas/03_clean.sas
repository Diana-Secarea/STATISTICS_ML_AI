/* ══════════════════════════════════════════════════════════════════════════════
 * Facilities 3, 5, 7 — DATA step: conditional processing, SAS functions, arrays
 *
 * Facility 3 — IF/THEN status labelling, DO loop for imputation
 * Facility 5 — LOG, MAX, SUM, CATX, ABS, ROUND + MDY, INTNX, QTR, DAY, DATEPART
 * Facility 7 — ARRAY over delay-cause columns; ARRAY seasonal benchmark lookup
 *
 * Input : WORK.FLIGHTS_SAMPLE
 * Output: WORK.FLIGHTS  (fully cleaned, enriched, formatted)
 * ══════════════════════════════════════════════════════════════════════════════ */

DATA WORK.FLIGHTS;
    SET WORK.FLIGHTS_SAMPLE;

    /* ── Facility 7a: Array — impute all five delay-cause columns to 0 ──────
     * NaN in BTS data means the delay type did not occur, not that it is unknown.
     * Mirrors Python:  df[delay_cause_cols].fillna(0)
     */
    ARRAY delay_causes[5] carrier_delay  weather_delay  nas_delay
                          security_delay late_aircraft_delay;
    DO i = 1 TO 5;
        IF delay_causes[i] = . THEN delay_causes[i] = 0;
    END;

    /* Count how many distinct delay types were active for this flight */
    n_active_causes = 0;
    DO i = 1 TO 5;
        IF delay_causes[i] > 0 THEN n_active_causes = n_active_causes + 1;
    END;
    DROP i;

    /* ── Facility 7b: Array — seasonal benchmark lookup table ─────────────
     * 12 expected avg departure delays (one per month) loaded from Python
     * analysis findings: Jul = 18.9 min peak, Oct = 4.3 min best month.
     * Using month as the subscript gives O(1) lookup — no scan needed.
     */
    ARRAY season_avg[12] _TEMPORARY_
        (8.2, 9.1, 10.3, 11.5, 13.2, 15.8,
         18.9, 17.4, 12.1,  4.3,  9.8, 11.2);

    IF 1 <= month <= 12 THEN DO;
        seasonal_benchmark = season_avg[month];
        delay_vs_seasonal  = ROUND(dep_delay - season_avg[month], 0.1);
        above_seasonal_avg = (dep_delay > season_avg[month]);
    END;

    /* ── Facility 3: Conditional processing ─────────────────────────────────*/

    /* Impute primary delay and cancellation code — mirrors handle_missing() */
    IF dep_delay         = .  THEN dep_delay         = 0;
    IF arr_delay         = .  THEN arr_delay         = 0;
    IF cancellation_code = '' THEN cancellation_code = 'N';

    /* Derive a flight status label — single pass over ordered conditions */
    IF      cancelled   = 1   THEN flight_status = 'Cancelled';
    ELSE IF dep_delay  >= 60  THEN flight_status = 'Severe';
    ELSE IF dep_delay  >= 16  THEN flight_status = 'Moderate';
    ELSE IF dep_delay  >  0   THEN flight_status = 'Minor';
    ELSE                           flight_status = 'On Time';

    /* Determine the single largest contributing delay cause */
    _max_c = MAX(carrier_delay, weather_delay,
                 nas_delay,     late_aircraft_delay);
    IF      _max_c <= 0                      THEN primary_cause = 'None';
    ELSE IF _max_c = carrier_delay           THEN primary_cause = 'Carrier';
    ELSE IF _max_c = weather_delay           THEN primary_cause = 'Weather';
    ELSE IF _max_c = nas_delay               THEN primary_cause = 'NAS / ATC';
    ELSE IF _max_c = late_aircraft_delay     THEN primary_cause = 'Late Aircraft';
    DROP _max_c;

    /* Weekend flag — matches Python: (day_of_week IN (6,7)) */
    is_weekend = (day_of_week IN (6, 7));

    /* ── Facility 5: SAS functions ───────────────────────────────────────────*/

    /* Log-transform arrival delay — identical shift logic to Python OLS step
     * shift ensures minimum value is 1 before LOG so result is always defined */
    _shift        = MAX(0, -arr_delay) + 1;
    log_arr_delay = LOG(arr_delay + _shift);
    DROP _shift;

    /* Aggregate delay causes — mirrors Python SUM() */
    total_cause_delay = SUM(carrier_delay, weather_delay,
                            nas_delay,     late_aircraft_delay);

    /* String functions */
    route           = CATX('-', STRIP(origin), STRIP(dest));  /* e.g. JFK-LAX */
    carrier_upper   = UPCASE(op_unique_carrier);

    /* Numeric utility measures */
    abs_arr_delay   = ABS(arr_delay);
    delay_per_mile  = ROUND(dep_delay / MAX(distance, 1), 0.001);
    on_time_flag    = (dep_delay <= 0 AND cancelled = 0);

    DROP carrier_upper;   /* kept only to show UPCASE; not needed downstream */

    /* ── Facility 5b: Date functions — MDY, INTNX, QTR, DAY, DATEPART ──────
     * BTS provides month as an integer (1–12). We reconstruct a full SAS date
     * from it so we can demonstrate the complete date-function suite without
     * depending on the FL_DATE column format from PROC IMPORT.
     */
    month_start   = MDY(month, 1, 2024);                         /* YYYY-MM-01        */
    month_end     = INTNX('MONTH', month_start, 0, 'END');        /* last day of month */
    next_month    = INTNX('MONTH', month_start, 1, 'BEGINNING');  /* 1st of next month */
    quarter       = QTR(month_start);                             /* 1 … 4            */
    days_in_month = DAY(month_end);                               /* 28 / 30 / 31     */
    is_peak_season = (quarter IN (2, 3));                         /* Q2–Q3 = summer    */

    /* DATEPART: recovers the date component from a SAS DATETIME value */
    _ref_dt    = DHMS(month_start, 12, 0, 0);   /* noon on the 1st of the month */
    month_date = DATEPART(_ref_dt);              /* strips the time → SAS date   */
    DROP _ref_dt;

    FORMAT month_start month_end next_month month_date DATE9.;

    /* ── Attach formats so all downstream PROCs inherit them ─────────────── */
    FORMAT dep_delay arr_delay   DELAY_SEV.
           op_unique_carrier     $CARRIER_FMT.
           month                 MONTH_FMT.
           day_of_week           DOW_FMT.
           cancellation_code     $CANCEL_FMT.;

RUN;

TITLE "Facility 3 / 5 / 7 — Cleaned dataset: core variables (first 10 rows)";
PROC PRINT DATA=WORK.FLIGHTS (OBS=10);
    VAR op_unique_carrier origin dest month dep_delay arr_delay
        flight_status primary_cause n_active_causes
        total_cause_delay log_arr_delay route;
RUN;

TITLE "Facility 5b — Date-derived variables (first 10 rows)";
PROC PRINT DATA=WORK.FLIGHTS (OBS=10);
    VAR month month_start month_end next_month quarter
        days_in_month is_peak_season
        seasonal_benchmark delay_vs_seasonal above_seasonal_avg;
RUN;

TITLE "Missing values remaining after imputation";
PROC MEANS DATA=WORK.FLIGHTS NMISS N;
    VAR dep_delay arr_delay carrier_delay weather_delay
        nas_delay security_delay late_aircraft_delay;
RUN;
TITLE;
