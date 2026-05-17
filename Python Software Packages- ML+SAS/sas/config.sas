/*
 * config.sas — global paths and session options.
 * Update DATA_DIR to the folder where your CSV files live in SAS Studio.
 * All other files read this via run_all.sas — do not %INCLUDE it twice.
 */

%LET DATA_DIR = /home/u64515957/data;
%LET OUT_DIR  = /home/u64515957/results;
%LET SEED     = 42;
%LET SAMPLE_N = 100000;

OPTIONS NODATE NONUMBER PAGESIZE=MAX LINESIZE=120 MERGENOBY=ERROR;
ODS GRAPHICS ON / WIDTH=960px HEIGHT=540px IMAGEFMT=PNG;
