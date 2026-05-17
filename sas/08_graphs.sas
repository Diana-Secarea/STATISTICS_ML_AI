/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 10 — Generating graphs (PROC SGPLOT, PROC SGPANEL)
 * Mirrors all six matplotlib charts from src/visualizations.py in SAS.
 *
 * Chart list:
 *   10a. Horizontal bar — avg arrival delay by carrier
 *   10b. Dual-line — monthly delay trend
 *   10c. Box plot — departure delay distribution by carrier
 *   10d. Stacked horizontal bar — delay causes per carrier
 *   10e. Scatter + LOESS — distance vs. arrival delay
 *   10f. Histogram + KDE — departure delay distribution
 *   10g. SGPANEL small multiples — arr delay per top-5 carrier
 *
 * Input : WORK.CARRIER_SUMMARY, WORK.MONTHLY_SUMMARY,
 *         WORK.FLIGHTS, WORK.CAUSE_SUMMARY
 * ══════════════════════════════════════════════════════════════════════════════ */

ODS GRAPHICS ON / WIDTH=960px HEIGHT=540px IMAGEFMT=PNG;

/* ── 10a. Horizontal bar — avg arrival delay by carrier ─────────────────── */
TITLE  "Facility 10a — Average Arrival Delay by Airline (2024)";
TITLE2 "Red = above zero (delayed); Green = below zero (early on average)";
PROC SGPLOT DATA=WORK.CARRIER_SUMMARY;
    HBAR carrier / RESPONSE    = avg_arr_delay
                   DATALABEL
                   DATALABELATTRS = (SIZE=8 WEIGHT=BOLD)
                   FILLATTRS   = (COLOR=SteelBlue)
                   OUTLINEATTRS= (COLOR=White);
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Black THICKNESS=1.5);
    XAXIS LABEL="Average Arrival Delay (minutes)" GRID GRIDATTRS=(COLOR=WhiteSmoke);
    YAXIS LABEL="Airline" VALUESFMT=$CARRIER_FMT. OFFSETMIN=0.05 OFFSETMAX=0.05;
    KEYLEGEND / NOBORDER;
RUN;
TITLE;

/* ── 10b. Dual-line — monthly delay trend ────────────────────────────────── */
TITLE  "Facility 10b — Monthly Delay Trend (2024)";
TITLE2 "Departure delay peaks in July/August; October is the best month";
PROC SGPLOT DATA=WORK.MONTHLY_SUMMARY;
    BAND   X=month LOWER=0 UPPER=avg_dep_delay /
               FILLATTRS=(COLOR=SteelBlue TRANSPARENCY=0.85)
               LEGENDLABEL="Dep Delay Area";
    SERIES X=month Y=avg_dep_delay /
               MARKERS
               LINEATTRS  =(COLOR=SteelBlue THICKNESS=2.5)
               MARKERATTRS=(SYMBOL=CircleFilled SIZE=8 COLOR=SteelBlue)
               LEGENDLABEL="Avg Departure Delay";
    SERIES X=month Y=avg_arr_delay /
               MARKERS
               LINEATTRS  =(COLOR=Crimson THICKNESS=2.5 PATTERN=ShortDash)
               MARKERATTRS=(SYMBOL=SquareFilled SIZE=7 COLOR=Crimson)
               LEGENDLABEL="Avg Arrival Delay";
    REFLINE 0 / AXIS=Y LINEATTRS=(COLOR=Gray PATTERN=Dot THICKNESS=1);
    XAXIS VALUES=(1 TO 12) VALUESFMT=MONTH_FMT.
          LABEL="Month" FITPOLICY=NONE;
    YAXIS LABEL="Average Delay (minutes)" GRID;
    KEYLEGEND / LOCATION=INSIDE POSITION=TOPRIGHT NOBORDER;
RUN;
TITLE;

/* ── 10c. Box plot — departure delay by carrier ──────────────────────────── */
TITLE  "Facility 10c — Departure Delay Distribution by Carrier";
TITLE2 "Width of box = IQR; whiskers = 1.5×IQR; dots = outliers";
PROC SGPLOT DATA=WORK.FLIGHTS;
    VBOX dep_delay / CATEGORY    = op_unique_carrier
                     FILLATTRS   = (COLOR=SteelBlue TRANSPARENCY=0.25)
                     LINEATTRS   = (COLOR=MidnightBlue)
                     WHISKERATTRS= (THICKNESS=1.5 COLOR=MidnightBlue)
                     MEANATTRS   = (SYMBOL=Diamond SIZE=6 COLOR=OrangeRed);
    REFLINE 0 / AXIS=Y LINEATTRS=(COLOR=Red PATTERN=ShortDash THICKNESS=1.5)
                LABEL="On-Time Threshold";
    YAXIS LABEL="Departure Delay (minutes)" MIN=-40 MAX=160 GRID;
    XAXIS LABEL="Airline" VALUESFMT=$CARRIER_FMT. FITPOLICY=ROTATE;
RUN;
TITLE;

/* ── 10d. Stacked horizontal bar — delay causes per carrier ────────────────
 * SGPLOT requires long (tidy) format: one row per carrier × cause.
 * Reshape CAUSE_SUMMARY from wide to long with a DATA step.
 */
DATA WORK.CAUSES_LONG;
    SET WORK.CAUSE_SUMMARY;
    LENGTH cause $20;
    cause = 'Carrier';       avg_minutes = avg_carrier;        OUTPUT;
    cause = 'Weather';       avg_minutes = avg_weather;        OUTPUT;
    cause = 'NAS / ATC';     avg_minutes = avg_nas;            OUTPUT;
    cause = 'Late Aircraft';  avg_minutes = avg_late_aircraft;  OUTPUT;
    KEEP carrier cause avg_minutes;
RUN;

TITLE  "Facility 10d — Delay Cause Breakdown by Airline";
TITLE2 "NAS/ATC and Late Aircraft dominate across all carriers";
PROC SGPLOT DATA=WORK.CAUSES_LONG;
    HBAR carrier / RESPONSE      = avg_minutes
                   GROUP         = cause
                   GROUPDISPLAY  = STACK
                   DATALABEL
                   DATALABELATTRS= (SIZE=7);
    XAXIS LABEL="Average Delay (minutes)" GRID;
    YAXIS LABEL="Airline" VALUESFMT=$CARRIER_FMT.;
    KEYLEGEND / TITLE="Delay Cause" LOCATION=OUTSIDE POSITION=RIGHT;
RUN;
TITLE;

/* ── 10e. Scatter + LOESS — distance vs. arrival delay ─────────────────────
 * Sample 5 000 rows so the scatter renders quickly — mirrors Python sample=5000
 */
PROC SURVEYSELECT DATA=WORK.FLIGHTS OUT=WORK.SCATTER_SAMPLE
    METHOD=SRS N=5000 SEED=&SEED NOPRINT;
RUN;

TITLE  "Facility 10e — Flight Distance vs. Arrival Delay";
TITLE2 "LOESS smoother shows the trend; most delay variation is independent of distance";
PROC SGPLOT DATA=WORK.SCATTER_SAMPLE;
    SCATTER X=distance Y=arr_delay /
                TRANSPARENCY = 0.65
                MARKERATTRS  = (SYMBOL=CircleFilled SIZE=4 COLOR=SlateGray);
    LOESS   X=distance Y=arr_delay /
                LINEATTRS    = (COLOR=OrangeRed THICKNESS=2.5)
                LEGENDLABEL  = "LOESS Trend";
    REFLINE 0 / AXIS=Y LINEATTRS=(COLOR=Red PATTERN=ShortDash THICKNESS=1)
                LABEL="On-Time";
    XAXIS LABEL="Distance (miles)" GRID;
    YAXIS LABEL="Arrival Delay (minutes)" GRID;
    KEYLEGEND / NOBORDER LOCATION=INSIDE POSITION=TOPRIGHT;
RUN;
TITLE;

/* ── 10f. Histogram + kernel density — departure delay distribution ──────── */
TITLE  "Facility 10f — Distribution of Departure Delay";
TITLE2 "Heavy right tail; most flights depart within ±15 minutes of schedule";
PROC SGPLOT DATA=WORK.FLIGHTS;
    HISTOGRAM dep_delay / BINWIDTH    = 5
                          FILLATTRS   = (COLOR=SteelBlue TRANSPARENCY=0.3)
                          LINEATTRS   = (COLOR=White);
    DENSITY   dep_delay / TYPE        = KERNEL
                          LINEATTRS   = (COLOR=DarkRed THICKNESS=2.5)
                          LEGENDLABEL = "Kernel Density";
    DENSITY   dep_delay / TYPE        = NORMAL
                          LINEATTRS   = (COLOR=ForestGreen THICKNESS=1.5
                                         PATTERN=ShortDash)
                          LEGENDLABEL = "Normal Fit";
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Red THICKNESS=1.5 PATTERN=ShortDash)
                LABEL="On-Time";
    XAXIS LABEL="Departure Delay (minutes)" MIN=-60 MAX=200 GRID;
    YAXIS LABEL="Density" GRID;
    KEYLEGEND / NOBORDER LOCATION=INSIDE POSITION=TOPRIGHT;
RUN;
TITLE;

/* ── 10g. SGPANEL small-multiples — arr delay for top-5 carriers ───────────
 * Mirrors Python: plot_delay_small_multiples()
 * First identify top-5 carriers by flight count, then panel on them.
 */
PROC SQL NOPRINT;
    SELECT op_unique_carrier INTO :top5_carriers SEPARATED BY ','
    FROM (
        SELECT op_unique_carrier, COUNT(*) AS n
        FROM WORK.FLIGHTS
        GROUP BY op_unique_carrier
        ORDER BY n DESC
    )
    WHERE MONOTONIC() <= 5;
QUIT;

DATA WORK.TOP5_FLIGHTS;
    SET WORK.FLIGHTS;
    WHERE op_unique_carrier IN (&top5_carriers.);
RUN;

TITLE  "Facility 10g — Arrival Delay Distribution: Top 5 Airlines (Small Multiples)";
PROC SGPANEL DATA=WORK.TOP5_FLIGHTS;
    PANELBY op_unique_carrier / COLUMNS=5 NOVARNAME
                                HEADERATTRS=(SIZE=10 WEIGHT=BOLD);
    HISTOGRAM arr_delay / BINWIDTH  = 8
                          FILLATTRS = (COLOR=SlateBlue TRANSPARENCY=0.25)
                          LINEATTRS = (COLOR=White);
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Red PATTERN=ShortDash THICKNESS=1);
    ROWAXIS  LABEL="Flights"         GRID;
    COLAXIS  LABEL="Arrival Delay (min)" MIN=-60 MAX=150;
    FORMAT op_unique_carrier $CARRIER_FMT.;
RUN;
TITLE;

ODS GRAPHICS OFF;
