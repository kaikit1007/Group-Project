import pandas as pd
url = "https://huggingface.co/datasets/jienweng/housing-prices-malaysia-2025/resolve/main/malaysia_house_price_data_2025.csv"

print(" connecting to the website...")
print("downloading the data from the website...\n")

try:
    # 1.download the whole data from website
    df = pd.read_csv(url)

    # 2. clean the data and turn all of it to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # 3. filter out the unlogic data
    df = df.dropna(subset=["median_price"])
    df = df[df["median_price"] > 0]

    # 4. save the data as csv file
    output_file = "malaysia_housing_2025.csv"
    df.to_csv(output_file, index=False)

    print("=" * 50)
    print(f"✅ Download Success！the data is saving：'{output_file}'")
    print("=" * 50)

    print("\n--- Top 5 data  ---")
    print(df.head(5))

except Exception as e:
    print(f"\n❌ Download Failed。Error: {e}")
    print(
        "💡 Hint:Check Your Wifi if u meet the problem of out of connection time! ")