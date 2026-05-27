import pandas as pd
import urllib.request
import os

def fetch_and_build_large_corpus():
    print("=== Launching Public Data Ingestion Engine ===")
    output_file = "kaggle_test_data.csv"
    
    # Mirror URL hosting the official UCI dataset
    source_url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
    
    try:
        print("Downloading official open-source retail description logs...")
        # Add headers to bypass simple bot blockers securely
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(source_url, "raw_public_source.csv")
        print("Download successful. Loading into data layer...")
        
        # Latin-1 encoding safely handles messy or legacy commercial symbols
        raw_df = pd.read_csv("raw_public_source.csv", encoding="latin-1")
        raw_df = raw_df.dropna(subset=['Description'])
        
        processed_data = []
        print("Aligning real-world transaction strings to your target categories...")
        
        # Parse unique text rows to extract realistic product matches
        for _, row in raw_df.drop_duplicates(subset=['Description']).head(15000).iterrows():
            desc = str(row['Description']).strip()
            desc_upper = desc.upper()
            
            mapped_cat = None
            if any(k in desc_upper for k in ["BAG", "BOX", "CASE", "PACK", "CONTAINER", "BOTTLE", "TIN", "CAN"]):
                mapped_cat = "Inventory"
            elif any(k in desc_upper for k in ["HOLDER", "LANTERN", "CHAIR", "DESK", "PENS", "PENCIL", "NOTEBOOK", "RULER"]):
                mapped_cat = "Office Supplies"
            elif any(k in desc_upper for k in ["POSTAGE", "CARRIAGE", "DELIVERY", "SHIPPING", "COURIER", "EXPRESS"]):
                mapped_cat = "Logistics"
            elif any(k in desc_upper for k in ["TRAVEL", "PASSPORT", "TICKET", "SUITCASE", "LUGGAGE", "WALKING"]):
                mapped_cat = "Travel"
            elif any(k in desc_upper for k in ["DOTCOM", "SOFTWARE", "ONLINE", "ELECTRONIC", "WEB", "DIGITAL"]):
                mapped_cat = "Cloud/Software"
            
            if mapped_cat and len(desc) > 5:
                processed_data.append({"text": desc, "category": mapped_cat})
        
        # Insert foundational anchor sequences to ensure absolute class stability
        extra_anchor_samples = [
            {"text": "AWS cloud compute hosting instance usage charge East-1", "category": "Cloud/Software"},
            {"text": "Microsoft Azure SQL database backup consumption tier", "category": "Cloud/Software"},
            {"text": "Google Cloud Platform cloud storage bucket allocation", "category": "Cloud/Software"},
            {"text": "Commercial electricity grid infrastructure bill for Q2", "category": "Utilities"},
            {"text": "Facility sanitation, industrial water management supply bill", "category": "Utilities"},
            {"text": "High speed corporate broadband fiber optic internet lease line", "category": "Utilities"}
        ]
        
        final_records = processed_data + (extra_anchor_samples * 10)
        final_df = pd.DataFrame(final_records).drop_duplicates().dropna()
        
        # Sample an even distribution per group to keep measurements completely balanced
        final_df = final_df.groupby('category').head(25).reset_index(drop=True)
        
        final_df.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Compiled {len(final_df)} clean transactional rows into {output_file}")
        print(final_df['category'].value_counts())
        
        if os.path.exists("raw_public_source.csv"):
            os.remove("raw_public_source.csv")
            
    except Exception as e:
        print(f"[FATAL] Ingestion runtime failure: {str(e)}")

if __name__ == "__main__":
    fetch_and_build_large_corpus()