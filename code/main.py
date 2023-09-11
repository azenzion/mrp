# Import the draft data
import os
import csv

csv_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "draft_data_public.LTR.PremierDraft.csv")

with open(csv_file_path, "r") as f:
    # Create a CSV reader
    csv_reader = csv.reader(f, delimiter=",")
    
    # Only look at the first 10 columns
    for row in csv_reader:
        print(row[:10])
        break

    exit()
    
    # Print the next 5 rows
    for i in range(45):  
        next_row = next(csv_reader)
        print(next_row[:10])
