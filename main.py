"""
Association Rule Mining on crop_production.csv
==============================================
Columns expected: State_Name, District_Name, Crop_Year, Season, Crop, Area, Production

Two analyses are run and saved to local disk (OUTPUT_DIR):

  A) Attribute rules      - every row is a transaction made of items like
                            "State_Name=Punjab", "Season=Rabi", "Crop=Wheat",
                            "Area_bin=High", "Yield_bin=Medium"
                            e.g. {Season=Rabi, State_Name=Punjab} -> {Crop=Wheat}

  B) Co-cultivation rules - one transaction = one (state, district, year, season)
                            and the items are the crops grown there
                            e.g. {Maize, Groundnut} -> {Sorghum}

Requirements:
    pip install pandas mlxtend matplotlib
Run:
    python crop_association_rules.py
    (in Colab: upload crop_production.csv, then set CSV_PATH = "/content/crop_production.csv")
"""

import os

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a display
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

# ----------------------------------------------------------------------------
# Settings (change these as needed)
# ----------------------------------------------------------------------------
CSV_PATH = "crop_production.csv"
OUTPUT_DIR = "association_results"   # created next to where you run the script

# Analysis A: attribute rules
A_MIN_SUPPORT = 0.02
A_MIN_CONFIDENCE = 0.60
A_MIN_LIFT = 1.2
A_MAX_LEN = 4

# Analysis B: co-cultivation rules
B_MIN_SUPPORT = 0.03
B_MIN_CONFIDENCE = 0.60
B_MIN_LIFT = 1.2
B_MAX_LEN = 4

TOP_N_PRINT = 15


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    # Season values have trailing spaces ("Kharif     "), so strip all text columns
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["Area", "Production"])   # ~3.7k rows lack Production
    df = df[df["Area"] > 0].copy()
    df["Yield"] = df["Production"] / df["Area"]
    print(f"Rows after cleaning: {len(df):,}")
    return df


def quantile_bin(series, labels=("Low", "Medium", "High")):
    """Split a numeric column into equal-sized Low/Medium/High groups."""
    return pd.qcut(series.rank(method="first"), q=len(labels), labels=list(labels))


def run_mining(onehot, min_support, min_confidence, min_lift, max_len):
    """Frequent itemsets -> rules, filtered by confidence and lift."""
    itemsets = fpgrowth(onehot, min_support=min_support,
                        use_colnames=True, max_len=max_len)
    print(f"  Frequent itemsets found: {len(itemsets):,}")
    if itemsets.empty:
        return itemsets, pd.DataFrame()

    try:
        rules = association_rules(itemsets, metric="confidence",
                                  min_threshold=min_confidence)
    except TypeError:
        # newer mlxtend versions also want the number of transactions
        rules = association_rules(itemsets, num_itemsets=len(onehot),
                                  metric="confidence",
                                  min_threshold=min_confidence)

    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    print(f"  Rules after filters (conf>={min_confidence}, lift>={min_lift}): {len(rules):,}")
    return itemsets, rules


def to_text(frozen):
    return ", ".join(sorted(frozen))


def make_saveable(itemsets, rules):
    """Convert frozensets to readable strings so they save cleanly to CSV."""
    itemsets = itemsets.copy()
    itemsets["itemsets"] = itemsets["itemsets"].apply(to_text)
    itemsets["length"] = itemsets["itemsets"].str.count(",") + 1
    itemsets = itemsets.sort_values("support", ascending=False)

    rules = rules.copy()
    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(to_text)
        rules["consequents"] = rules["consequents"].apply(to_text)
    return itemsets, rules


def save_plot(rules, title, path):
    if rules.empty:
        return
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(rules["support"], rules["confidence"],
                     c=rules["lift"], cmap="viridis", alpha=0.7, s=25)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_and_report(name, itemsets, rules):
    itemsets, rules = make_saveable(itemsets, rules)
    itemsets.to_csv(os.path.join(OUTPUT_DIR, f"{name}_frequent_itemsets.csv"), index=False)
    rules.to_csv(os.path.join(OUTPUT_DIR, f"{name}_rules.csv"), index=False)
    save_plot(rules, f"{name}: support vs confidence (colour = lift)",
              os.path.join(OUTPUT_DIR, f"{name}_rules_plot.png"))

    print(f"\n  Top {TOP_N_PRINT} rules by lift:")
    if rules.empty:
        print("  (none - try lowering min_support / min_confidence / min_lift)")
    else:
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(rules[cols].head(TOP_N_PRINT).round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# Analysis A: attribute rules (one row = one transaction)
# ----------------------------------------------------------------------------
def analysis_attributes(df):
    print("\n=== A) Attribute rules ===")
    items = pd.DataFrame({
        "State_Name": df["State_Name"],
        "Season": df["Season"],
        "Crop": df["Crop"],
        "Area_bin": quantile_bin(df["Area"]),
        "Yield_bin": quantile_bin(df["Yield"]),
    })
    # one-hot columns look like "Crop=Rice", "Season=Kharif", ...
    onehot = pd.get_dummies(items.astype(str), prefix_sep="=", dtype=bool)
    print(f"  Transactions: {len(onehot):,}   Items: {onehot.shape[1]}")

    itemsets, rules = run_mining(onehot, A_MIN_SUPPORT, A_MIN_CONFIDENCE,
                                 A_MIN_LIFT, A_MAX_LEN)
    save_and_report("A_attributes", itemsets, rules)


# ----------------------------------------------------------------------------
# Analysis B: co-cultivation rules (basket = state + district + year + season)
# ----------------------------------------------------------------------------
def analysis_cocultivation(df):
    print("\n=== B) Co-cultivation rules ===")
    basket_keys = ["State_Name", "District_Name", "Crop_Year", "Season"]
    onehot = pd.crosstab([df[k] for k in basket_keys], df["Crop"]).gt(0)
    print(f"  Baskets: {len(onehot):,}   Crops: {onehot.shape[1]}")

    # keep only baskets with 2+ crops (single-crop baskets cannot form rules)
    onehot = onehot[onehot.sum(axis=1) >= 2]
    print(f"  Baskets with 2+ crops: {len(onehot):,}")

    itemsets, rules = run_mining(onehot, B_MIN_SUPPORT, B_MIN_CONFIDENCE,
                                 B_MIN_LIFT, B_MAX_LEN)
    save_and_report("B_cocultivation", itemsets, rules)


# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_clean(CSV_PATH)
    analysis_attributes(df)
    analysis_cocultivation(df)
    print(f"\nDone. Results saved in: {os.path.abspath(OUTPUT_DIR)}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("  -", f)


if __name__ == "__main__":
    main()
