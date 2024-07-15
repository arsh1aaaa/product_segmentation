# API CODE (first 2 functions work, identify outliers doesnt work)

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse 

app = FastAPI()

def load_and_prepare_data(file: UploadFile, sheet_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp): 
    # Read the data source
    df = pd.read_excel(file.file, sheet_name=sheet_name)
    # Replace all NaN/- values with 0
    df = df.fillna(0)
    # Convert columns to datetime if necessary
    df.columns = [df.columns[0], df.columns[1]] + [pd.to_datetime(col) for col in df.columns[2:]]
    # Filter the DataFrame to include only the specified date range + work on a copy of the filtered df (data integrity)
    segmentation_period_df = df[['Item Code', 'Item Name'] + [col for col in df.columns[2:] if start_date <= col <= end_date]].copy()
    return segmentation_period_df

def abc_analysis(segmentation_period_df: pd.DataFrame, a_threshold: int, b_threshold: int): 
    # Sum values row-wise excluding the first two columns
    row_sums = segmentation_period_df.iloc[:, 2:].sum(axis=1)
    segmentation_period_df['Sales Value'] = row_sums

    summarised_df = segmentation_period_df[['Item Code', 'Item Name', 'Sales Value']].copy()
    summarised_df.sort_values(by='Sales Value', ascending=False, inplace=True)

    total_revenue = summarised_df['Sales Value'].sum()
    num_items = len(summarised_df)

    summarised_df['Cumulative Sales Value'] = summarised_df['Sales Value'].cumsum()
    summarised_df['% Rev'] = (summarised_df['Cumulative Sales Value'] / total_revenue) * 100

    summarised_df['Rank'] = summarised_df.index + 1
    summarised_df['% Items'] = (summarised_df['Rank'] / num_items) * 100

    # Classify into categories A, B, and C
    abc_bins = [0, a_threshold, b_threshold, 100]
    abc_labels = ['A', 'B', 'C']
    summarised_df['ABC'] = pd.cut(summarised_df['% Rev'], bins=abc_bins, labels=abc_labels, right=True, include_lowest=True)
    return summarised_df

# def identify_outliers(segmentation_period_df: pd.DataFrame, summarised_df: pd.DataFrame):
#     # Calculate the sales standard deviation and mean
#     sales_data_columns = segmentation_period_df.columns[2:-1]
#     segmentation_period_df['Sales Std Dev'] = segmentation_period_df[sales_data_columns].std(axis=1)
#     segmentation_period_df['Sales Mean'] = segmentation_period_df[sales_data_columns].mean(axis=1)
    
#     # Merge these values into summarised_df
#     summarised_df['Sales Std Dev'] = segmentation_period_df['Sales Std Dev']
#     summarised_df['Sales Mean'] = segmentation_period_df['Sales Mean']

#     # Calculate the coefficient of variation
#     summarised_df['coefficient_variation_%'] = (summarised_df['Sales Std Dev'] / summarised_df['Sales Mean']) * 100

#     # Handle NaN values: replace with a large number or some other logic
#     summarised_df['coefficient_variation_%'].fillna(1000000, inplace=True)
    

def identify_outliers(segmentation_period_df, summarised_df):  #works
    # Calculate the sales standard deviation and mean
    sales_data_columns = segmentation_period_df.columns[2:-1]
    segmentation_period_df['Sales Std Dev'] = segmentation_period_df[sales_data_columns].std(axis=1)
    segmentation_period_df['Sales Mean'] = segmentation_period_df[sales_data_columns].mean(axis=1)
    
    # Merge these values into summarised_df
    summarised_df['Sales Std Dev'] = segmentation_period_df['Sales Std Dev']
    summarised_df['Sales Mean'] = segmentation_period_df['Sales Mean']

    # Calculate the coefficient of variation
    summarised_df['coefficient_variation_%'] = (summarised_df['Sales Std Dev'] / summarised_df['Sales Mean']) * 100

    # Check for NaN values in coefficient_variation_%
    nan_values = summarised_df[summarised_df['coefficient_variation_%'].isna()]

    # Handle NaN values: replace with a large number or some other logic
    summarised_df['coefficient_variation_%'].fillna(1000000, inplace=True)

    # Find values outside expected range
    negative_values = summarised_df[summarised_df['coefficient_variation_%'] < 0]
    excessive_values = summarised_df[summarised_df['coefficient_variation_%'] > 1000000]

    return {
        'nan_values': nan_values,
        'negative_values': negative_values,
        'excessive_values': excessive_values
    }


def xyz_analysis(summarised_df: pd.DataFrame, x_threshold: int, y_threshold: int):
    # Ensure the coefficient_variation_% column exists
    if 'coefficient_variation_%' not in summarised_df.columns:
        raise KeyError("The 'coefficient_variation_%' column is missing in the DataFrame.")

    # Classify into categories X, Y, Z
    xyz_bins = [0, x_threshold, y_threshold, 1000000]
    xyz_labels = ['X', 'Y', 'Z']
    summarised_df['XYZ'] = pd.cut(summarised_df['coefficient_variation_%'], bins=xyz_bins, labels=xyz_labels, right=True, include_lowest=True)

    # Combine ABC and XYZ classifications
    summarised_df['ABCXYZ'] = summarised_df['ABC'].astype(str) + summarised_df['XYZ'].astype(str)

    # Define the custom sort order
    custom_sort_order = ['AX', 'AY', 'BX', 'AZ', 'BY', 'CX', 'BZ', 'CY', 'CZ']

    # Create a custom sort column based on the custom order
    summarised_df['sort_order'] = summarised_df['ABCXYZ'].apply(lambda x: custom_sort_order.index(x) if x in custom_sort_order else len(custom_sort_order))

    # Sort the DataFrame based on the custom sort order
    summarised_df = summarised_df.sort_values('sort_order').drop(columns=['sort_order'])

    return summarised_df

def abc_xyz_summary(summarised_df):  # Distribution summary of number of items in each category
    num_items = len(summarised_df)
    abcxyz_counts = summarised_df['ABCXYZ'].value_counts().sort_index()
    abcxyz_percentages = (abcxyz_counts / num_items) * 100
    summary = {
            "category_counts": abcxyz_counts.to_dict(),
            "category_percentages": abcxyz_percentages.to_dict()
        }
    return summary

def get_excel_data(completed_df): 
    df = pd.DataFrame(completed_df)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    return StreamingResponse(
        BytesIO(buffer.getvalue()),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename=data.csv"}
)

@app.post("/abcxyz-analysis/")
async def abcxyz_analysis(file: UploadFile = File(...), sheet_name: str = 'Unprocessed Data', start_date: str = '2021-06-01', end_date: str = '2022-06-01', a_threshold: int = 20, b_threshold: int = 40, x_threshold: int = 15, y_threshold: int = 35):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Load and prepare data
    segmentation_period_df = load_and_prepare_data(file, sheet_name, start_date, end_date)

    # Perform ABC analysis
    summarised_df = abc_analysis(segmentation_period_df, a_threshold, b_threshold)

    # Identify outliers and add coefficient of variation
    identify_outliers(segmentation_period_df, summarised_df)

    # Perform XYZ analysis
    completed_df = xyz_analysis(summarised_df, x_threshold, y_threshold)

    # CHOOSE THE RETURN STATEMENT DEPENDING ON DESIRED DATA FORMAT (EITHER CSV OR RAW JSON FILE)
    # return completed_df.to_dict(orient="records")
    return get_excel_data(completed_df)

@app.post("/abcxyz-summary/")
async def abcxyz_summary(file: UploadFile = File(...), sheet_name: str = 'Unprocessed Data', start_date: str = '2021-06-01', end_date: str = '2022-06-01', a_threshold: int = 20, b_threshold: int = 40, x_threshold: int = 15, y_threshold: int = 35):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Load and prepare data
    segmentation_period_df = load_and_prepare_data(file, sheet_name, start_date, end_date)

    # Perform ABC analysis
    summarised_df = abc_analysis(segmentation_period_df, a_threshold, b_threshold)

    # Identify outliers and add coefficient of variation
    identify_outliers(segmentation_period_df, summarised_df)

    # Perform XYZ analysis
    completed_df = xyz_analysis(summarised_df, x_threshold, y_threshold)

    # Get ABC-XYZ summary                    
    summary = abc_xyz_summary(completed_df)                       
    
    return summary                                              

# To run the app, save this script and run: uvicorn main:app --reload
                                                                                                                                             
                                                                                                                                                 
                                                                                                                                                     
                                                                                                                                                                                                            