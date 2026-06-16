import pandas as pd
import os


def extract_excel_to_csv():
    excel_file = "Jadual Penerbitan MHPI Q1 2026P_3Jun.xlsx"
    output_csv = "Final_Training_Data.csv"

    print(f"Extracting all sheets from {excel_file}...")

    # Load all sheets from the Excel file
    xls = pd.read_excel(excel_file, sheet_name=None)

    # Concatenate all sheets into one single DataFrame
    all_data = pd.concat(xls.values(), ignore_index=True)

    # Save as CSV
    all_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Successfully created {output_csv}")


if __name__ == "__main__":
    extract_excel_to_csv()