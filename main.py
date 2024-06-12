from preprocessing import load_records
from process_records import process_dataset

def main():

    # Manually set the start index and number of records to process
    start_index = 10000  # Set your desired starting index here
    num_records = 100  # Set your desired number of records to process here

    # Get records
    records = load_records(r'C:\Users\20190896\Downloads\Thesis\Notebooks\combined_records.txt')

    # Ensure the number of records to process does not exceed the length of records
    end_index = min(start_index + num_records, len(records))
    selected_records = records[start_index:end_index]

    # Process dataset
    process_dataset(selected_records, start_index)

if __name__ == "__main__":
    main()
