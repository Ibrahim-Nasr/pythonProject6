import wfdb
import pandas as pd
from config import database_name, required_sigs, req_seg_duration, batch_size, save_interval
from preprocessing import preprocess_signals, output_dir
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

processed_count = 0  # Counter for processed records
save_count = 0  # Counter for save intervals

def process_record(record, index):
    global processed_count
    processed_count += 1
    print(f'Processing Record at index {index}: {record}', end="")
    record_path = record.split('/')
    record_dir = f'{database_name}/{"/".join(record_path[:-1])}'
    record_name = record_path[-1]
    try:
        record_data = wfdb.rdrecord(record_name, pn_dir=record_dir)
        seg_length = record_data.sig_len / record_data.fs
        if seg_length < req_seg_duration:
            print(f' (too short at {seg_length / 60:.1f} mins)')
            return None
        sigs_present = record_data.sig_name
        if all(x in sigs_present for x in required_sigs):
            print(' (met requirements)')
            success, error_message = preprocess_signals(record_name, record_dir, output_dir)
            if success:
                return {'dir': record_dir, 'seg_name': record_name, 'length': seg_length}
            else:
                print(f' (preprocessing failed: {error_message})')
                return None
        else:
            print(' (long enough, but missing signal(s))')
            return None
    except Exception as e:
        print(f' Error processing record: {e}')
        return None

def process_batch(batch, start_index):
    results = []
    for i, record in enumerate(batch):
        result = process_record(record, start_index + i)
        if result:
            results.append(result)
    return results

def save_results(results, processed_count):
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f'matching_records_up_to_{processed_count}.csv', index=False)
        print(f"Saved results to matching_records_up_to_{processed_count}.csv")

def process_dataset(records, start_index):
    global save_count
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    current_results = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_batch, batch, start_index + idx * batch_size) for idx, batch in enumerate(batches)]
        for future in as_completed(futures):
            batch_results = future.result()
            current_results.extend(batch_results)

            # Save results every save_interval records processed
            if processed_count >= save_count * save_interval:
                save_count += 1  # Increment save count
                save_results(current_results, processed_count)
                current_results = []  # Clear the results after saving

    # Save any remaining results
    if current_results:
        save_results(current_results, processed_count)

    print("Processing complete.")
