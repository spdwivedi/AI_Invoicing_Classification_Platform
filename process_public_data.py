import pandas as pd
import sys
import os

def normalize_external_csv(input_path, output_name="kaggle_test_data.csv"):
    if not os.path.exists(input_path):
        print(f"Error: Target raw file not found at {input_path}")
        return

    print(f"Reading raw data matrix from: {input_path}...")
    # Load raw data - handles fallback encoding options for system logs
    try:
        raw_df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(input_path, encoding="latin-1")

    print("Columns discovered in target file:", list(raw_df.columns))

    # --- ARCHITECTURAL STEP: DYNAMIC SOURCE IDENTIFICATION ---
    # Automatically map varying source column names to our internal variables
    text_col = None
    category_col = None

    # Common text description column variants
    for col in ['Description', 'item_description', 'Commodity', 'Product Name', 'text', 'title']:
        if col in raw_df.columns:
            text_col = col
            break
            
    # Common category/department label column variants
    for col in ['Category', 'Department', 'object', 'character', 'category', 'class']:
        if col in raw_df.columns:
            category_col = col
            break

    if not text_col or not category_col:
        print("\n[CRITICAL] Could not map column layouts automatically.")
        print(f"Please update script strings to match your file columns.")
        return

    print(f"Mapping Columns -> Text: '{text_col}', Label: '{category_col}'")

    # Drop null rows to keep the evaluation clean
    clean_df = raw_df[[text_col, category_col]].dropna().copy()
    clean_df.columns = ['raw_text', 'raw_label']

    # --- ARCHITECTURAL STEP: CATEGORY ALIGNMENT MATRIX ---
    # Map raw strings from external sets directly into our 6 targets
    alignment_map = {
        # Cloud/Software Mappings
        'TECHNOLOGY': 'Cloud/Software', 'SOFTWARE': 'Cloud/Software', 'SaaS': 'Cloud/Software',
        # Office Supplies Mappings
        'OFFICE PRODUCTS': 'Office Supplies', 'STATIONERY': 'Office Supplies', 'FURNITURE': 'Office Supplies',
        # Logistics Mappings
        'SHIPPING': 'Logistics', 'COURIER SERVICES': 'Logistics', 'TRANSPORTATION': 'Logistics',
        # Utilities Mappings
        'WATER & POWER': 'Utilities', 'ELECTRICITY': 'Utilities', 'TELECOMMUNICATIONS': 'Utilities',
        # Travel Mappings
        'TRAVEL & ACCOMMODATION': 'Travel', 'TRANSPORTATION-PASSENGER': 'Travel',
        # Inventory Mappings
        'RAW MATERIALS': 'Inventory', 'MANUFACTURING STOCK': 'Inventory', 'BULK COMPONENTS': 'Inventory'
    }

    # Normalize incoming strings to uppercase for consistent matching
    clean_df['normalized_label'] = clean_df['raw_label'].astype(str).str.upper()
    
    # Apply matching logic
    processed_records = []
    for _, row in clean_df.iterrows():
        mapped_class = None
        for pattern, target_class in alignment_map.items():
            if pattern in row['normalized_label']:
                mapped_class = target_class
                break
        
        if mapped_class:
            processed_records.append({
                'text': row['raw_text'],
                'category': mapped_class
            })

    if not processed_records:
        print("Warning: No records matched the mapping dictionary keys.")
        return

    final_df = pd.DataFrame(processed_records)
    
    # Optional: Balance sample sizes across categories to keep evaluations unbiased
    final_df = final_df.groupby('category').head(40).reset_index(drop=True)

    final_df.to_csv(output_name, index=False)
    print(f"\nProcessing complete! Cleaned dataset saved at: {output_name}")
    print(final_df['category'].value_counts())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_public_data.py <path_to_raw_kaggle_csv>")
    else:
        normalize_external_csv(sys.argv[1])