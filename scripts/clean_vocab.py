import os

def clean_and_split(input_file, output_file):
    cleaned = set()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty or nan lines
            if not line or line.lower() == "nan":
                continue

            # Split on comma and take the first part
            primary = line.split(",")[0].strip()
            cleaned.add(primary)

    # Write sorted unique results
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sorted(cleaned):
            f.write(item + "\n")

    print(f"Saved {len(cleaned)} unique cleaned entries to {output_file}")

if __name__ == "__main__":
    for name in ["cleaned_unique_specialties", "cleaned_capabilities", "cleaned_unique_departments",
                 "cleaned_unique_hospitals", "cleaned_facility_types"]:
        clean_and_split(str(VOCAB / "raw" / f"{name}.txt"), str(VOCAB / f"{name}.txt"))
