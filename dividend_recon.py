import pandas as pd
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

def llm_column_mapping(nbim_columns, custody_columns,client, MODEL) -> dict:
    prompt = f"""
        You are a data mapping expert. I have two CSV files containing dividend bookings data with different column naming conventions.

        **Task:** Analyze the column headers from both files and create a mapping that identifies which columns represent the same data, even though they have different names.

        **NBIM Dividend Data Booking Headers:**
        {nbim_columns}

        **Custody Dividend Data Booking Headers:**
        {custody_columns}

        **Instructions:**
        1. Compare the column headers from both files
        2. Identify columns that represent the same data despite different naming
        3. Consider common variations like:
        - Different naming conventions (camelCase vs snake_case vs spaces)
        - Abbreviations vs full names (e.g., "Div Amount" vs "Dividend Amount")
        - Different word orders (e.g., "Payment Date" vs "Date of Payment")
        - Synonyms (e.g., "Security" vs "Instrument" vs "Asset")

        **Output Format:**
        **Output Format:**
        Provide ONLY a valid JSON object with the mapping (no code blocks, no markdown):

        {{
            "column_map": [
                ["Custody_column_name 1", "NBIM_column_name 1"],
                ["Custody_column_name 2", "NBIM_column_name 2"]
            ],
            "unmatched_nbim": ["col1", "col2"],
            "unmatched_custody": ["col3", "col4"]
        }}
           
    """
    try: 
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        result = json.loads(text)

    except json.JSONDecodeError as e:
        result = {
            "error": f"JSON parsing error: {str(e)}",
            "raw_response": text
        }
    except Exception as e:
        result = {
            "error": f"Error: {str(e)}"
        }

    return result



def get_event_keys(nbim, custody):
    """Get unique event keys from both datasets"""
    nbim_keys = set(nbim['COAC_EVENT_KEY'].unique())
    custody_keys = set(custody['COAC_EVENT_KEY'].unique())
    all_keys = sorted(list(nbim_keys.union(custody_keys)))
    
    return all_keys

def event_check(event_key, column_map, nbim, custody):
    nbim = nbim[nbim['COAC_EVENT_KEY'] == event_key]
    custody = custody[custody['COAC_EVENT_KEY'] == event_key]

    if len(nbim) > 1 or len(custody) > 1:
        return {}
    
    context = {}
    # deviation = []
    for i in column_map:
        tekst = i[0]+"/"+i[1]
        fields = [nbim.iloc[0][i[1]], custody.iloc[0][i[0]]]
        context[tekst]=fields


    return context

def llm_reconciliation(context, client, MODEL):
    """
    Bruker OpenAI API til å analysere reconciliation breaks
    
    Args:
        context: Dict med reconciliation data
        client: OpenAI() klient
    
    Returns:
        Dict med LLM analyse
    """
    prompt = f"""You are analyzing a dividend reconciliation break for NBIM (Norway's sovereign wealth fund).

DATA:
{json.dumps(context, indent=2)}

ANALYZE THIS BREAK:

1. WHY did this happen? Consider:
   - Securities lending (shares on loan reduce custody position)
   - Tax rate differences (treaty rates vs standard withholding)
   - Payment date timing differences
   - FX rate timing
   - Data entry errors

2. SEVERITY: Classify as critical, high, medium, low, or no_break
   - critical: >5% difference OR missing records
   - high: 1-5% difference
   - medium: 0.1-1% difference  
   - low: <0.1% difference
   - no_break: perfect match

3. CAN THIS BE AUTO-FIXED SAFELY?
   - Only if it's a known pattern with high confidence
   - Never auto-fix if >$100K or critical severity

RESPOND IN JSON:
{{
    "severity": "critical|high|medium|low|no_break",
    "explanation": "One clear sentence explaining what happened",
    "root_cause": "The underlying reason",
    "recommended_action": "What to do about it",
    "auto_remediable": true/false,
    "confidence": 0.0-1.0
}}

Be concise and specific."""
    
    try: 
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"} 

        )
        text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        result = json.loads(text)

    except json.JSONDecodeError as e:
        result = {
            "severity": "error",
            "error": f"JSON parsing error: {str(e)}",
            "raw_response": text,
            "explanation": "Failed to parse LLM response",
            "root_cause": "API error",
            "recommended_action": "Retry or investigate manually",
            "auto_remediable": False,
            "confidence": 0.0
        }
    except Exception as e:
        result = {
            "severity": "error",
            "error": f"API error: {str(e)}",
            "explanation": "LLM API call failed",
            "root_cause": "Connection or API error",
            "recommended_action": "Check API key and retry",
            "auto_remediable": False,
            "confidence": 0.0
        }

    return result

def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    MODEL = "gpt-4o-mini"
    INPUT_NBIM = "resources/NBIM_Dividend_Bookings.csv"
    INPUT_CUSTODY = "resources/CUSTODY_Dividend_Bookings.csv"

    client = OpenAI(api_key=OPENAI_API_KEY)

    nbim = pd.read_csv(INPUT_NBIM, sep=";")
    custody = pd.read_csv(INPUT_CUSTODY, sep=";")


    #Bruke LLM til å lage mapping mellom kolonnenavnene
    custody_columns = custody.columns
    nbim_columns = nbim.columns

    #svar fra llm-en
    #mapping = {'column_map': [['ISIN', 'ISIN'], ['SEDOL', 'SEDOL'], ['CUSTODIAN', 'CUSTODIAN'], ['NOMINAL_BASIS', 'NOMINAL_BASIS'], ['EVENT_EX_DATE', 'EXDATE'], ['EVENT_PAYMENT_DATE', 'PAYMENT_DATE'], ['DIV_RATE', 'DIVIDENDS_PER_SHARE'], ['GROSS_AMOUNT', 'GROSS_AMOUNT_QUOTATION'], ['NET_AMOUNT_QC', 'NET_AMOUNT_QUOTATION'], ['NET_AMOUNT_SC', 'NET_AMOUNT_SETTLEMENT'], ['TAX_RATE', 'WTHTAX_RATE'], ['CURRENCIES', 'QUOTATION_CURRENCY'], ['BANK_ACCOUNTS', 'BANK_ACCOUNT'], ['SETTLED_CURRENCY', 'SETTLEMENT_CURRENCY']], 'unmatched_nbim': ['COAC_EVENT_KEY', 'INSTRUMENT_DESCRIPTION', 'TICKER', 'ORGANISATION_NAME', 'AVG_FX_RATE_QUOTATION_TO_PORTFOLIO', 'GROSS_AMOUNT_PORTFOLIO', 'NET_AMOUNT_PORTFOLIO', 'WTHTAX_COST_QUOTATION', 'WTHTAX_COST_SETTLEMENT', 'WTHTAX_COST_PORTFOLIO', 'LOCALTAX_COST_QUOTATION', 'LOCALTAX_COST_SETTLEMENT', 'TOTAL_TAX_RATE', 'EXRESPRDIV_COST_QUOTATION', 'EXRESPRDIV_COST_SETTLEMENT', 'RESTITUTION_RATE'], 'unmatched_custody': ['EVENT_TYPE', 'LOAN_QUANTITY', 'HOLDING_QUANTITY', 'LENDING_PERCENTAGE', 'RECORD_DATE', 'PAY_DATE', 'IS_CROSS_CURRENCY_REVERSAL', 'FX_RATE', 'POSSIBLE_RESTITUTION_PAYMENT', 'POSSIBLE_RESTITUTION_AMOUNT', 'ADR_FEE', 'ADR_FEE_RATE']}
    mapping = llm_column_mapping(nbim_columns, custody_columns,client,MODEL)
    column_map = mapping['column_map']

    allkeys = get_event_keys(nbim, custody)
    

    for key in allkeys:
        context = event_check(key, column_map, nbim, custody)
        if len(context)>0:
            clean_context = {
                k: [float(v) if isinstance(v, (np.float64, np.int64)) else v for v in val]
                for k, val in context.items()
                }
            resultat = llm_reconciliation(clean_context, client, MODEL)
            print(resultat)
        
if __name__ == "__main__":
    main() 
