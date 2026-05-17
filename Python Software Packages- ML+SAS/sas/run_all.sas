/*
 * run_all.sas — master pipeline runner.
 *
 * HOW TO USE IN SAS ONDEMAND FOR ACADEMICS:
 *   1. Upload flight_data_2024.csv and Airports-Only.csv to your SAS Studio file area.
 *   2. Edit DATA_DIR and OUT_DIR in config.sas to match the upload path.
 *   3. Open this file in SAS Studio and click Run (F3).
 *
 * Each step writes its output to WORK.* tables that the next step consumes.
 * Running this file is the only entry point — do not run the numbered files alone.
 */

%INCLUDE "config.sas";
%INCLUDE "01_import.sas";
%INCLUDE "02_formats.sas";
%INCLUDE "03_clean.sas";
%INCLUDE "04_subsets.sas";
%INCLUDE "05_merge.sas";
%INCLUDE "06_statistics.sas";
%INCLUDE "07_report.sas";
%INCLUDE "08_graphs.sas";
%INCLUDE "09_ml.sas";
