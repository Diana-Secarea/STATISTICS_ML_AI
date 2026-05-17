/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 9 — Statistical procedures
 * PROC MEANS, PROC FREQ, PROC CORR, PROC UNIVARIATE, PROC GLM
 *
 * Input : WORK.FLIGHTS, WORK.CARRIER_SUMMARY
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── 9a. Descriptive statistics — mirrors Python .describe() ─────────────── */
TITLE  "Facility 9a — Descriptive Statistics: Delay Variables";
TITLE2 "N, missing, mean, std, min, quartiles, max across 100 k flights";
PROC MEANS DATA=WORK.FLIGHTS
    N NMISS MEAN STD MIN P25 MEDIAN P75 MAX MAXDEC=2;
    VAR dep_delay arr_delay distance air_time
        carrier_delay weather_delay nas_delay
        security_delay late_aircraft_delay
        total_cause_delay log_arr_delay;
RUN;
TITLE;

/* ── 9b. Grouped means by carrier — mirrors Python groupby + .describe() ─── */
TITLE  "Facility 9b — Mean dep_delay and arr_delay by Carrier";
PROC MEANS DATA=WORK.FLIGHTS MEAN MEDIAN STD N MAXDEC=2 NWAY;
    CLASS op_unique_carrier;
    VAR dep_delay arr_delay;
    FORMAT op_unique_carrier $CARRIER_FMT.;
RUN;
TITLE;

/* ── 9c. Frequency tables ─────────────────────────────────────────────────── */
TITLE "Facility 9c — Flight Status Distribution";
PROC FREQ DATA=WORK.FLIGHTS ORDER=FREQ;
    TABLES flight_status / NOCUM;
RUN;
TITLE;

TITLE "Facility 9c — Cancellation Code Breakdown";
PROC FREQ DATA=WORK.FLIGHTS ORDER=FREQ;
    TABLES cancellation_code / NOCUM;
    FORMAT cancellation_code $CANCEL_FMT.;
RUN;
TITLE;

TITLE "Facility 9c — Flight Volume and Cancellation Rate by Carrier";
PROC FREQ DATA=WORK.FLIGHTS ORDER=FREQ;
    TABLES op_unique_carrier * cancelled / NOCOL NOPERCENT;
    FORMAT op_unique_carrier $CARRIER_FMT.;
RUN;
TITLE;

TITLE "Facility 9c — Primary Delay Cause Frequency";
PROC FREQ DATA=WORK.FLIGHTS ORDER=FREQ;
    TABLES primary_cause / NOCUM;
    WHERE primary_cause NE 'None';
RUN;
TITLE;

/* ── 9d. Correlation matrix — mirrors Python df.corr() ──────────────────── */
TITLE  "Facility 9d — Pearson Correlations: Delay Variables";
TITLE2 "Key question: are delay types correlated? Does distance predict delay?";
PROC CORR DATA=WORK.FLIGHTS NOSIMPLE PLOTS=MATRIX(HISTOGRAM);
    VAR dep_delay arr_delay carrier_delay weather_delay
        nas_delay late_aircraft_delay distance air_time;
RUN;
TITLE;

/* ── 9e. Distribution analysis — normality tests ─────────────────────────── */
TITLE  "Facility 9e — PROC UNIVARIATE: Departure Delay Distribution";
TITLE2 "Skewness, kurtosis, normality tests for dep_delay";
PROC UNIVARIATE DATA=WORK.FLIGHTS NORMAL PLOT;
    VAR dep_delay;
    HISTOGRAM dep_delay / NORMAL(COLOR=RED) KERNEL(COLOR=BLUE);
    INSET N MEAN STD SKEWNESS KURTOSIS / POSITION=NE;
RUN;
TITLE;

/* ── 9f. One-way ANOVA — is mean delay significantly different across carriers? */
TITLE  "Facility 9f — One-Way ANOVA: Departure Delay by Carrier";
TITLE2 "H0: mean dep_delay is equal across all airlines";
PROC GLM DATA=WORK.FLIGHTS PLOTS=NONE;
    CLASS op_unique_carrier;
    MODEL dep_delay = op_unique_carrier;
    FORMAT op_unique_carrier $CARRIER_FMT.;
RUN;
QUIT;
TITLE;
