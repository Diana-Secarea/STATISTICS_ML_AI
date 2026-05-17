/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 11 — SAS Machine Learning (base SAS — no Viya required)
 *
 * Three models that mirror the Python src/models.py exactly:
 *
 *   11a. PROC FASTCLUS  — k-means clustering → delay tiers  (mirrors KMeans)
 *   11b. PROC LOGISTIC  — binary classification             (mirrors LogisticRegression)
 *   11c. PROC REG       — simple OLS for quick coefficient  (mirrors statsmodels OLS)
 *        PROC GLM       — full OLS with carrier dummies     (mirrors statsmodels OLS)
 *
 * Input : WORK.FLIGHTS
 * Output: WORK.CLUSTERED, WORK.CLUSTER_PROFILE,
 *         WORK.LR_PRED,   WORK.ROC_DATA,
 *         WORK.REG_PRED,  WORK.GLM_ESTIMATES
 * ══════════════════════════════════════════════════════════════════════════════ */

ODS GRAPHICS ON / WIDTH=900px HEIGHT=500px IMAGEFMT=PNG;


/* ══════════════════════════════════════════════════════════════════════════════
 * 11a — PROC FASTCLUS: K-Means style clustering into 4 delay tiers
 *
 * Features: dep_delay, arr_delay, distance, air_time
 * Features must be standardised first — PROC STDIZE mirrors Python StandardScaler
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Remove rows with missing cluster features */
DATA WORK.FLIGHTS_FOR_CLUS;
    SET WORK.FLIGHTS;
    WHERE dep_delay NE . AND arr_delay NE .
      AND distance  NE . AND air_time  NE .
      AND cancelled = 0;           /* cluster only operated flights */
RUN;

/* Standardise (zero mean, unit variance) — identical to Python StandardScaler */
PROC STDIZE DATA=WORK.FLIGHTS_FOR_CLUS
            OUT =WORK.FLIGHTS_STD
            METHOD=STD REPLACE;
    VAR dep_delay arr_delay distance air_time;
RUN;

/* Run FASTCLUS with k=4, matching Python KMeans(n_clusters=4, random_state=42) */
TITLE  "Facility 11a — PROC FASTCLUS: K-Means Clustering (k = 4)";
TITLE2 "Flights segmented into 4 delay tiers based on dep_delay, arr_delay, distance, air_time";
PROC FASTCLUS DATA   = WORK.FLIGHTS_STD
              OUT    = WORK.CLUSTERED_STD
              MAXCLU = 4
              MAXITER= 100
              SEED   = 42
              NOPRINT;                    /* suppress raw centroid dump */
    VAR dep_delay arr_delay distance air_time;
RUN;
TITLE;

/* Attach the cluster label back to the original (unstandardised) data */
DATA WORK.CLUSTERED;
    MERGE WORK.FLIGHTS_FOR_CLUS
          WORK.CLUSTERED_STD (KEEP=CLUSTER);
RUN;

/* Profile each cluster on original scale — mirrors cluster_summary() */
TITLE  "Facility 11a — Cluster Centroids on Original Scale";
TITLE2 "Cluster with lowest arr_delay = Tier 0 (best); highest = Tier 3 (worst)";
PROC MEANS DATA=WORK.CLUSTERED MEAN MEDIAN N MAXDEC=2 NWAY;
    CLASS CLUSTER;
    VAR dep_delay arr_delay distance air_time;
    OUTPUT OUT=WORK.CLUSTER_PROFILE MEAN= / AUTONAME;
RUN;
TITLE;

/* Scatter plot: dep_delay vs arr_delay coloured by cluster — mirrors plot_clusters() */
TITLE "Facility 11a — Cluster Scatter: Departure vs Arrival Delay";
PROC SGPLOT DATA=WORK.CLUSTERED (WHERE=(dep_delay BETWEEN -30 AND 180));
    SCATTER X=dep_delay Y=arr_delay /
                GROUP        = CLUSTER
                TRANSPARENCY = 0.6
                MARKERATTRS  = (SYMBOL=CircleFilled SIZE=4);
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Gray PATTERN=Dot);
    REFLINE 0 / AXIS=Y LINEATTRS=(COLOR=Gray PATTERN=Dot);
    XAXIS LABEL="Departure Delay (min)" GRID;
    YAXIS LABEL="Arrival Delay (min)"   GRID;
    KEYLEGEND / TITLE="Cluster (Delay Tier)" NOBORDER;
RUN;
TITLE;

/* Cluster size breakdown */
TITLE "Facility 11a — Cluster Membership Counts";
PROC FREQ DATA=WORK.CLUSTERED ORDER=FORMATTED;
    TABLES CLUSTER / NOCUM;
RUN;
TITLE;


/* ══════════════════════════════════════════════════════════════════════════════
 * 11b — PROC LOGISTIC: Predict flight cancellation (binary outcome)
 *
 * Target : cancelled (1 = cancelled, 0 = operated)
 * Features: dep_delay, distance, carrier_delay, weather_delay,
 *            nas_delay, late_aircraft_delay, month, day_of_week
 *
 * NOTE: air_time is intentionally EXCLUDED.
 *       Cancelled flights have air_time imputed to 0 → data leakage.
 *       A model using air_time would achieve AUC ~0.999 by detecting
 *       the imputation pattern, not real cancellation signal.
 *
 * Class imbalance (~1.4% cancelled): handled via PRIOR statement.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* 80/20 stratified split — mirrors train_test_split(stratify=y) */
PROC SURVEYSELECT DATA=WORK.FLIGHTS
                  OUT =WORK.LR_SPLIT
                  METHOD=SRS SAMPRATE=0.8 SEED=42 OUTALL NOPRINT;
    STRATA cancelled;       /* stratify on target to preserve class ratio */
RUN;

DATA WORK.LR_TRAIN WORK.LR_TEST;
    SET WORK.LR_SPLIT;
    IF Selected = 1 THEN OUTPUT WORK.LR_TRAIN;
    ELSE                  OUTPUT WORK.LR_TEST;
RUN;

TITLE  "Facility 11b — PROC LOGISTIC: Predicting Flight Cancellation";
TITLE2 "Target: cancelled=1 (~1.4%); class imbalance handled via PRIOR";
PROC LOGISTIC DATA    = WORK.LR_TRAIN
              OUTMODEL= WORK.LR_MODEL
              PLOTS(ONLY) = (ROC EFFECT);
    MODEL cancelled(EVENT='1') =
              dep_delay distance
              carrier_delay weather_delay nas_delay late_aircraft_delay
              month day_of_week
          / LINK    = LOGIT
            SELECTION= NONE
            OUTROC   = WORK.ROC_DATA
            CTABLE               /* classification table at p=0.5 */
            LACKFIT;             /* Hosmer-Lemeshow goodness-of-fit */
    /* Weight rare class to mirror class_weight='balanced' */
    PRIOR cancelled / EQUALPRIOR;
    OUTPUT OUT=WORK.LR_TRAIN_PRED PREDICTED=pred_prob;
RUN;
TITLE;

/* Score the held-out test set */
PROC LOGISTIC INMODEL=WORK.LR_MODEL;
    SCORE DATA=WORK.LR_TEST OUT=WORK.LR_PRED (RENAME=(P_1=pred_prob));
RUN;

/* Compute AUC on test set */
TITLE "Facility 11b — AUC on Hold-Out Test Set (20%)";
PROC LOGISTIC DATA=WORK.LR_PRED PLOTS=NONE;
    MODEL cancelled(EVENT='1') = pred_prob / NOFIT OUTROC=WORK.ROC_TEST;
RUN;
TITLE;

/* Manual ROC curve plot on test data */
DATA WORK.ROC_TEST;
    SET WORK.ROC_TEST;
    _sensitivity_ = _SENSIT_;
    _fpr_         = 1 - _1MSPEC_;
RUN;

TITLE "Facility 11b — ROC Curve: Cancellation Prediction (Test Set)";
PROC SGPLOT DATA=WORK.ROC_TEST;
    SERIES  X=_fpr_ Y=_sensitivity_ /
                LINEATTRS   = (COLOR=SteelBlue THICKNESS=2.5)
                LEGENDLABEL = "Logistic Regression";
    LINEPARM X=0 Y=0 SLOPE=1 /
                LINEATTRS   = (COLOR=Gray PATTERN=ShortDash THICKNESS=1)
                LEGENDLABEL = "Random Classifier";
    XAXIS LABEL="False Positive Rate" VALUES=(0 TO 1 BY 0.1) GRID;
    YAXIS LABEL="True Positive Rate"  VALUES=(0 TO 1 BY 0.1) GRID;
    KEYLEGEND / NOBORDER LOCATION=INSIDE POSITION=BOTTOMRIGHT;
RUN;
TITLE;

/* Feature coefficient plot — mirrors lr_feature_importance() */
PROC SQL NOPRINT;
    CREATE TABLE WORK.LR_COEFS AS
    SELECT Variable, Estimate, ProbChiSq AS pvalue
    FROM   WORK.LR_MODEL.ParameterEstimates
    WHERE  Variable NE 'Intercept'
    ORDER BY ABS(Estimate) DESC;
QUIT;

TITLE "Facility 11b — Logistic Regression Coefficients (sorted by |effect|)";
PROC SGPLOT DATA=WORK.LR_COEFS;
    HBAR Variable / RESPONSE     = Estimate
                    COLORRESPONSE = pvalue
                    COLORMODEL    = (SteelBlue LightGray)
                    DATALABEL
                    DATALABELATTRS=(SIZE=8);
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Black THICKNESS=1);
    XAXIS LABEL="Log-Odds Coefficient" GRID;
    YAXIS LABEL="Feature";
    GRADLEGEND / TITLE="p-value";
RUN;
TITLE;


/* ══════════════════════════════════════════════════════════════════════════════
 * 11c — PROC REG + PROC GLM: OLS regression — explain arrival delay
 *
 * Target : log_arr_delay  (log-transformed to reduce skew — same as Python)
 * PROC REG  : numeric features only — simple, interpretable baseline
 * PROC GLM  : adds carrier CLASS variable → full model with airline dummies
 *             mirrors Python OLS with pd.get_dummies(op_unique_carrier)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Drop rows with any missing on model variables */
DATA WORK.OLS_DATA;
    SET WORK.FLIGHTS;
    WHERE log_arr_delay NE .
      AND dep_delay     NE . AND distance         NE .
      AND carrier_delay NE . AND weather_delay     NE .
      AND nas_delay     NE . AND late_aircraft_delay NE .
      AND month         NE . AND day_of_week       NE .;
RUN;

/* ── Simple PROC REG (numeric features only) ─────────────────────────────── */
TITLE  "Facility 11c — PROC REG: OLS Regression — Arrival Delay";
TITLE2 "Target: log(arr_delay + offset) | Numeric features only";
PROC REG DATA=WORK.OLS_DATA
         PLOTS(ONLY)=(RESIDUALS RESIDUALPLOT QQ);
    MODEL log_arr_delay =
              dep_delay distance
              carrier_delay weather_delay nas_delay late_aircraft_delay
              month day_of_week
          / STB          /* standardised beta coefficients */
            VIF          /* variance inflation — check multicollinearity */
            SELECTION=NONE;
    OUTPUT OUT=WORK.REG_PRED
           PREDICTED = yhat
           RESIDUAL  = resid
           STUDENT   = student_resid;
RUN;
QUIT;
TITLE;

/* ── Full PROC GLM with carrier CLASS variable ───────────────────────────── */
TITLE  "Facility 11c — PROC GLM: OLS with Carrier Fixed Effects";
TITLE2 "Carrier dummies reveal structural delay differences across airlines";
PROC GLM DATA =WORK.OLS_DATA
         PLOTS=(DIAGNOSTICS RESIDUALS);
    CLASS op_unique_carrier;
    MODEL log_arr_delay =
              dep_delay distance
              carrier_delay weather_delay nas_delay late_aircraft_delay
              month day_of_week
              op_unique_carrier     /* airline fixed effects */
          / SOLUTION               /* print all coefficient estimates */
            SS3;                   /* Type III tests — standard in SAS */
    FORMAT op_unique_carrier $CARRIER_FMT.;
    OUTPUT OUT=WORK.GLM_PRED
           PREDICTED = yhat
           RESIDUAL  = resid;
    ODS OUTPUT ParameterEstimates = WORK.GLM_ESTIMATES;
RUN;
QUIT;
TITLE;

/* Coefficient plot for GLM — mirrors plot_ols_coefficients() */
PROC SQL NOPRINT;
    CREATE TABLE WORK.GLM_COEF_PLOT AS
    SELECT Parameter, Estimate, Probt AS pvalue
    FROM   WORK.GLM_ESTIMATES
    WHERE  Parameter NE 'Intercept'
      AND  Estimate  NE .
    ORDER BY ABS(Estimate) DESC;
QUIT;

DATA WORK.GLM_COEF_TOP15;
    SET WORK.GLM_COEF_PLOT;
    IF _N_ <= 15;      /* top 15 by absolute coefficient — matches Python top_n=15 */
RUN;

TITLE "Facility 11c — GLM Coefficient Plot: Top 15 Features by |Estimate|";
PROC SGPLOT DATA=WORK.GLM_COEF_TOP15;
    HBAR Parameter / RESPONSE     = Estimate
                     FILLATTRS    = (COLOR=SteelBlue)
                     DATALABEL
                     DATALABELATTRS=(SIZE=8 WEIGHT=BOLD);
    REFLINE 0 / AXIS=X LINEATTRS=(COLOR=Black THICKNESS=1);
    XAXIS LABEL="OLS Coefficient (log scale)" GRID;
    YAXIS LABEL="Feature";
RUN;
TITLE;

/* Residual plot — mirrors plot_ols_residuals() */
PROC SURVEYSELECT DATA=WORK.GLM_PRED OUT=WORK.RESID_SAMPLE
    METHOD=SRS N=3000 SEED=42 NOPRINT;
RUN;

TITLE  "Facility 11c — OLS Residual Plot (3 000-row sample)";
TITLE2 "Random scatter around 0 = model assumptions satisfied";
PROC SGPLOT DATA=WORK.RESID_SAMPLE;
    SCATTER X=yhat Y=resid /
                TRANSPARENCY = 0.55
                MARKERATTRS  = (SYMBOL=CircleFilled SIZE=4 COLOR=DarkSlateGray);
    REFLINE 0 / AXIS=Y LINEATTRS=(COLOR=Red PATTERN=ShortDash THICKNESS=1.5)
                LABEL="Zero Line";
    LOESS X=yhat Y=resid /
                LINEATTRS=(COLOR=OrangeRed THICKNESS=2)
                LEGENDLABEL="LOESS trend (should be flat)";
    XAXIS LABEL="Fitted Values (log arr_delay)" GRID;
    YAXIS LABEL="Residuals"                     GRID;
    KEYLEGEND / NOBORDER LOCATION=INSIDE POSITION=TOPRIGHT;
RUN;
TITLE;

ODS GRAPHICS OFF;
