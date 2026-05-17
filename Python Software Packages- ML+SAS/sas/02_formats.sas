/* ══════════════════════════════════════════════════════════════════════════════
 * Facility 2 — User-defined formats (PROC FORMAT)
 * Defines reusable label mappings applied throughout the pipeline.
 * ══════════════════════════════════════════════════════════════════════════════ */

PROC FORMAT;

    /* Airline IATA code → full name */
    VALUE $CARRIER_FMT
        'AA' = 'American Airlines'
        'AS' = 'Alaska Airlines'
        'B6' = 'JetBlue Airways'
        'DL' = 'Delta Air Lines'
        'F9' = 'Frontier Airlines'
        'G4' = 'Allegiant Air'
        'HA' = 'Hawaiian Airlines'
        'NK' = 'Spirit Airlines'
        'OH' = 'PSA Airlines'
        'OO' = 'SkyWest Airlines'
        'QX' = 'Horizon Air'
        'UA' = 'United Airlines'
        'WN' = 'Southwest Airlines'
        'YV' = 'Mesa Airlines'
        'YX' = 'Republic Airways'
        OTHER = 'Other';

    /* Numeric departure delay → severity band */
    VALUE DELAY_SEV
        LOW  -< 0   = 'Early'
        0    -< 16  = 'On Time / Minor'
        16   -< 60  = 'Moderate'
        60   - HIGH = 'Severe';

    /* Month number → abbreviated name */
    VALUE MONTH_FMT
        1  = 'Jan'   2  = 'Feb'   3  = 'Mar'
        4  = 'Apr'   5  = 'May'   6  = 'Jun'
        7  = 'Jul'   8  = 'Aug'   9  = 'Sep'
        10 = 'Oct'   11 = 'Nov'   12 = 'Dec';

    /* BTS cancellation code → reason */
    VALUE $CANCEL_FMT
        'A' = 'Carrier Issue'
        'B' = 'Weather'
        'C' = 'NAS / ATC'
        'D' = 'Security'
        'N' = 'Not Cancelled';

    /* ISO day-of-week → label (BTS: 1=Monday … 7=Sunday) */
    VALUE DOW_FMT
        1 = 'Monday'    2 = 'Tuesday'   3 = 'Wednesday'
        4 = 'Thursday'  5 = 'Friday'    6 = 'Saturday'
        7 = 'Sunday';

    /* Price-band equivalent — delay tier label for reports */
    VALUE $STATUS_FMT
        'On Time'   = 'On Time'
        'Minor'     = 'Minor   (1–15 min)'
        'Moderate'  = 'Moderate (16–59 min)'
        'Severe'    = 'Severe  (60+ min)'
        'Cancelled' = 'Cancelled';

RUN;

TITLE "Facility 2 — User-defined formats registered";
PROC FORMAT FMTLIB LIB=WORK; RUN;
TITLE;
