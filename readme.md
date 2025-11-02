# Dividend Reconciliation Tool

System that uses LLM to compare dividend booking data between NBIM and custody record.

## Overview

This tool automates the reconciliation of dividend payments by:

- Using AI to map columns between different data formats
- Identifying discrepancies in dividend bookings
- Analyzing reconciliation breaks with severity classification
- Providing actionable recommendations for resolution

## Features

- **Column Mapping**: Uses GPT-4o-mini to automatically map columns between NBIM and custody data formats, handling variations in naming conventions
- **Event-Based Reconciliation**: Compares dividend events using COAC_EVENT_KEY as the unique identifier
- **AI-Powered Break Analysis**: Analyzes discrepancies to identify root causes.
- **Severity Classification**: Categorizes breaks as critical, high, medium, low, or no_break based on materiality
- **Auto-Remediation Assessment**: Determines if breaks can be safely auto-corrected

## Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages:
  - pandas
  - openai
  - python-dotenv
  - numpy

## Installation

1. Clone the repository and navigate to the project directory

2. Install required packages:
```bash
pip install pandas openai python-dotenv numpy
```

3. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Ensure your CSV files are in the `resources/` directory:
   - `NBIM_Dividend_Bookings.csv`
   - `CUSTODY_Dividend_Bookings.csv`

## Usage

Run the reconciliation:

```bash
python dividend_recon.py
```

The script will:

1. Load both CSV files
2. Use LLM to create column mappings
3. Iterate through all event keys
4. Compare matching events
5. Output reconciliation analysis for each break



## Key Functions

### `llm_column_mapping()`
Maps column names between NBIM and custody formats using GPT-4o-mini, handling:
- Different naming conventions (camelCase, snake_case, spaces)
- Abbreviations vs full names
- Different word orders
- Synonyms

### `get_event_keys()`
Extracts unique COAC_EVENT_KEY values from both datasets for comparison.

### `event_check()`
Compares a single event across both datasets, creating a context dictionary with matched column values.
(still missing column values only in one of the datasets)

### `llm_reconciliation()`
Analyzes reconciliation breaks using LLM to:
- Identify root causes
- Classify severity
- Recommend actions
- Assess auto-remediation feasibility

## Error Handling

The tool includes comprehensive error handling for:
- JSON parsing errors from LLM responses
- API connection failures
- Missing or malformed data
- Invalid event keys

## Limitations

- Currently only processes events with single records (skips multi-record events)
- Requires COAC_EVENT_KEY to be present in both datasets
- Context sent to llm is not complete

## Future Enhancements

Potential improvements:
- Support for multi-record event reconciliation
- Better user interface for visualization of results
- Logging
- See if some events can be analyzed without using llm for all of them
- General improvement of code
