
from typing import Tuple, List
import pandera as pa
from pandera import Column, DataFrameSchema, Check


def validate_data(df):
    # Standardize Gender to prevent case issues
    #df["Gender"] = df["Gender"].astype(str).str.strip().str.upper()
    schema = DataFrameSchema(
        {
            "AGE": Column(float, Check.in_range(0,150)),
            "Gender": Column(str, Check.isin(["M","F"])),

            "Urea": Column(float, Check.in_range(0,50)),
            "Cr": Column(float, Check.in_range(0,800)),
            "HbA1c": Column(float, Check.in_range(0,20)),
            "Chol": Column(float, Check.in_range(0,15)),
            "TG": Column(float, Check.in_range(0,15)),
            "HDL": Column(float, Check.in_range(0,15)),
            "LDL": Column(float, Check.in_range(0,15)),
            "VLDL": Column(float, Check.in_range(0,45)),
            "BMI": Column(float, Check.in_range(11,59)),
        },
        coerce=False
    )
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        # Get column names only if any rows actually failed
        if "column" in err.failure_cases.columns:
            failed_columns = err.failure_cases["column"].dropna().unique().tolist()

    is_valid = len(failed_columns) == 0

    if is_valid:
        print("Data validation passed.")
    else:
        print("Data validation detected issues in columns:", failed_columns)

    return df, is_valid, failed_columns
"""
    try:
        # Run validation but don't crash on failure
        schema.validate(df, lazy=True)
        print("Data validation passed.")
        return df, True, []
    except pa.errors.SchemaErrors as err:
        # err.failure_cases contains the failing rows
        failed_columns = err.failure_cases["column"].dropna().unique().tolist()
        print("Data validation detected issues:", failed_columns)

        # Example fix: impute or correct values
        #if "Gender" in failed_columns:
            # Replace invalid values with NaN for later processing
            #df.loc[~df["Gender"].isin(["M","F"]), "Gender"] = None
        
        #if "AGE" in failed_columns:
            # Replace invalid AGE values with median
            #df["AGE"] = df["AGE"].fillna(df["AGE"].median())

        #if "Urea" in failed_columns:
            # Replace invalid AGE values with median
           # df["Urea"] = df["Urea"].fillna(df["Urea"].median())
        
        #if "Cr" in failed_columns:
            # Replace invalid AGE values with median
            #df["Cr"] = df["Cr"].fillna(df["Cr"].median())
        
        #if "HbA1c" in failed_columns:
            # Replace invalid AGE values with median
            #df["HbA1c"] = df["HbA1c"].fillna(df["HbA1c"].median())
        
        #if "Chol" in failed_columns:
            # Replace invalid AGE values with median
            #df["Chol"] = df["Chol"].fillna(df["Chol"].median())

        #if "TG" in failed_columns:
            # Replace invalid AGE values with median
            #df["TG"] = df["TG"].fillna(df["TG"].median())

        #if "HDL" in failed_columns:
            # Replace invalid AGE values with median
            #df["HDL"] = df["HDL"].fillna(df["HDL"].median())

        #if "LDL" in failed_columns:
            # Replace invalid AGE values with median
           # df["LDL"] = df["LDL"].fillna(df["LDL"].median())

        #if "VLDL" in failed_columns:
            # Replace invalid AGE values with median
            #df["VLDL"] = df["VLDL"].fillna(df["VLDL"].median())

        # Add other column fixes here as needed """

        #print("All issues Handled during Preprocessing. Pipeline will continue.")
        # return df, False, failed_columns 