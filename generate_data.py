import pandas as pd

# Multi-class baseline sample matrix
seed_data = [
    {"text": "Blue Dart courier charges for warehouse delivery", "category": "Logistics"},
    {"text": "DHL express shipping international freight delivery fees", "category": "Logistics"},
    {"text": "FedEx cargo parcel transit shipping fee receipt", "category": "Logistics"},
    {"text": "A4 printing paper sheets, notebooks, and binders pack", "category": "Office Supplies"},
    {"text": "Ballpoint pens, white board markers, and staplers pack", "category": "Office Supplies"},
    {"text": "Ergonomic adjustable office desk chairs and furniture", "category": "Office Supplies"},
    {"text": "AWS monthly cloud hosting subscription server bill", "category": "Cloud/Software"},
    {"text": "Google Cloud Platform processing instance usage invoice", "category": "Cloud/Software"},
    {"text": "Microsoft Azure database storage compute instance costs", "category": "Cloud/Software"},
    {"text": "Electricity grid power distribution consumption bill corporate", "category": "Utilities"},
    {"text": "Commercial water facility sanitation consumption charges", "category": "Utilities"},
    {"text": "High speed fiber broadband internet line lease contract", "category": "Utilities"},
    {"text": "Business flight tickets booking round trip economy", "category": "Travel"},
    {"text": "Corporate luxury hotel accommodation suite reservation", "category": "Travel"},
    {"text": "Uber rides expense allowance validation for client visit", "category": "Travel"},
    {"text": "Raw materials bulk steel and plastic warehouse procurement", "category": "Inventory"},
    {"text": "Finished goods restocking product order for factory shelf", "category": "Inventory"},
    {"text": "Wholesale safety equipment items stocking distribution pipeline", "category": "Inventory"}
]

# Multiply dataset entries to establish structural volume for model fit
df = pd.DataFrame(seed_data * 10)
df.to_csv("data.csv", index=False)
print(f"Dataset successfully compiled! Created data.csv with {len(df)} lines.")