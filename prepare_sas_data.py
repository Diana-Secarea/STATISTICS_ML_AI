import pandas as pd
import os

TARGET_BYTES = int(0.95 * 1024 ** 3)  # 950 MB — just under the 1 GB limit
INPUT  = "data/flight_data_2024.csv"
OUTPUT = "data/flight_data_2024.csv"
TEMP   = OUTPUT + ".tmp"
CHUNK_SIZE = 50_000

first_chunk = True
total_rows = 0

with open(TEMP, "w") as f_out:
    for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE):
        chunk.to_csv(f_out, index=False, header=first_chunk)
        first_chunk = False
        total_rows += len(chunk)
        current_size = os.path.getsize(TEMP)
        print(f"Rows: {total_rows:,} | Size: {current_size / 1024**3:.3f} GB", end="\r")
        if current_size >= TARGET_BYTES:
            break

os.replace(TEMP, OUTPUT)
final_size = os.path.getsize(OUTPUT) / 1024**2
print(f"\nDone. Rows written: {total_rows:,} | Final size: {final_size:.1f} MB")
