import pandas as pd
import random
from faker import Faker

fake = Faker()

# Define multi-layered contextual maps for realistic variations
templates = {
    "Logistics": [
        "Courier charges from {company} for {item} transit",
        "International shipping invoice by {company} ref #{num}",
        "Freight delivery fees for bulk transport of {item}",
        "Express priority parcel handling costs by {company}"
    ],
    "Office Supplies": [
        "Purchase of {item} and stationery items from {company}",
        "Supply run: ergonomic {item} and desk organization gear",
        "Bulk corporate printing paper, notebooks, and pens pack",
        "Office utilities upgrade: {company} commercial desk furniture"
    ],
    "Cloud/Software": [
        "Monthly cloud compute instance bill for {company} servers",
        "SaaS production platform database subscription charge for {company}",
        "Enterprise API hosting platform integration tier fee",
        "License renewal for centralized project management tool"
    ],
    "Utilities": [
        "Commercial electricity grid infrastructure bill for Q2",
        "Facility sanitation, industrial water management supply bill",
        "High speed corporate broadband fiber optic internet lease line",
        "Municipal power grid distribution consumption statement"
    ],
    "Travel": [
        "Business travel: Round-trip flight tickets to {city} for client review",
        "Corporate luxury hotel accommodation suite booking for executive summit",
        "On-demand rideshare allowance reimbursement for stakeholder site visit",
        "Airport car rental transit package for regional sales meetup"
    ],
    "Inventory": [
        "Bulk procurement invoice for raw {item} manufacturing stock",
        "Factory fulfillment center restocking: wholesale product components",
        "Distribution warehouse pipeline order for commercial grade materials",
        "Raw component supplier batch import for assembly line execution"
    ]
}

items_pool = {
    "Logistics": ["heavy machinery", "finished units", "retail cargo", "electronic parts"],
    "Office Supplies": ["adjustable chairs", "whiteboards", "staplers", "binder clips"],
    "Inventory": ["structural steel coils", "polymer pellets", "circuit boards", "aluminum sheets"]
}

def build_synthetic_set(num_samples_per_class=25):
    synthetic_records = []
    
    for category, phrase_list in templates.items():
        for _ in range(num_samples_per_class):
            phrase = random.choice(phrase_list)
            
            # Dynamically inject context variables
            text_line = phrase.format(
                company=fake.company(),
                item=random.choice(items_pool.get(category, ["materials"])),
                num=fake.random_number(digits=6),
                city=fake.city()
            )
            synthetic_records.append({"text": text_line, "category": category})
            
    df_synthetic = pd.DataFrame(synthetic_records)
    # Shuffle dataset rows to test sequence independence
    df_synthetic = df_synthetic.sample(frac=1).reset_index(drop=True)
    
    output_filename = "faker_test_data.csv"
    df_synthetic.to_csv(output_filename, index=False)
    print(f"Success! Generated {len(df_synthetic)} diverse test records at: {output_filename}")

if __name__ == "__main__":
    build_synthetic_set(30) # Generates a balanced matrix of 180 unseen items