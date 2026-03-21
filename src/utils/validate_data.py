"""
Data validation script. 
"""

import great_expectations as ge
from typing import Tuple, List

def validate_data (df) -> Tuple[bool, List[str]]:
    """Validate the dataset using Great Expectations."""
    print ("Validating data schema and required columns...")
    
    ge_df = ge.from_pandas(df)
    # core demographic features 
    ge_df.expect_columns_to_exist (['AGE', 'Gender'])
    ge_df.expect_column_values_to_not_be_null(['AGE', 'Gender'])

    # core service features 
    ge_df.expect_columns_to_exist (['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI'])
    ge_df.expect_column_values_to_not_be_null(['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']) 

    # === LOGIC VALIDATION ===
    print ("Validating data logic constraints....")

    # Gender must be one of expected values (data integrity check)
    ge_df.expect_column_values_to_be_in_set('Gender', ['M', 'F'])

    # === NUMERICAL RANGE VALIDATION ===
    print ("Validating numerical ranges and logic constraints for features...")

    ge_df.expect_column_values_to_be_between('AGE', min_value=0, max_value=150)
    ge_df.expect_column_values_to_be_between('Urea', min_value=0, max_value=50)
    ge_df.expect_column_values_to_be_between('Cr', min_value=0, max_value=800)
    ge_df.expect_column_values_to_be_between('HbA1c', min_value=0, max_value=20)
    ge_df.expect_column_values_to_be_between('Chol', min_value=0, max_value=15)
    ge_df.expect_column_values_to_be_between('TG', min_value=0, max_value=15)
    ge_df.expect_column_values_to_be_between('HDL', min_value=0, max_value=15)
    ge_df.expect_column_values_to_be_between('LDL', min_value=0, max_value=15)
    ge_df.expect_column_values_to_be_between('VLDL', min_value=0, max_value=45)
    ge_df.expect_column_values_to_be_between('BMI', min_value=11, max_value=59)

    # === RUN VALIDATION SUITE ===
    print (" Running validation suite...")
    results = ge_df.validate()

    # === PROCESS VALIDATION RESULTS ===

    failed_expectations = []
    for r in results['results']:
        if not r['success']:
            expectation_type = r['expectation_config']['expectation_type']
            failed_expectations.append('expectation_type')

    # Print validation results summary
    total_checks = len(results['results'])
    passed_checks = sum(1 for r in results['results'] if r['success']) 
    failed_checks = total_checks - passed_checks

    if results['success']:
        print (f'Data validation passed: {passed_checks}/{total_checks} checks successful.')
    else:
        print (f'Data validation failed: {failed_checks}/{total_checks} checks failed.')
        print (f'Failed expectations: {failed_expectations}')
        
    return results['success'], failed_expectations
