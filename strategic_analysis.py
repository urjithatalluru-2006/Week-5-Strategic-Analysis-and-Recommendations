import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("automobile.csv.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------------------
# Visualization 1
# Top 10 Most Expensive Cars
# -------------------------------

top_cars = df.nlargest(10, "price")[["CarName", "price"]]

plt.figure(figsize=(10,6))
sns.barplot(
    data=top_cars,
    x="price",
    y="CarName"
)

plt.title("Top 10 Most Expensive Cars")
plt.xlabel("Price")
plt.ylabel("Car Name")

plt.tight_layout()
plt.savefig("top_expensive_cars.png")
plt.show()

# -------------------------------
# Visualization 2
# Average Price by Fuel Type
# -------------------------------

plt.figure(figsize=(8,5))

df.groupby("fueltype")["price"].mean().plot(
    kind="bar"
)

plt.title("Average Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price")

plt.tight_layout()
plt.savefig("price_by_fueltype.png")
plt.show()

# -------------------------------
# Visualization 3
# Horsepower vs Price
# -------------------------------

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="horsepower",
    y="price"
)

plt.title("Horsepower vs Price")

plt.tight_layout()
plt.savefig("horsepower_price_strategy.png")
plt.show()

# -------------------------------
# Visualization 4
# Average Price by Car Body
# -------------------------------

plt.figure(figsize=(8,5))

df.groupby("carbody")["price"].mean().sort_values().plot(
    kind="bar"
)

plt.title("Average Price by Car Body Type")
plt.ylabel("Average Price")

plt.tight_layout()
plt.savefig("carbody_price_analysis.png")
plt.show()

# -------------------------------
# Visualization 5
# Correlation Heatmap
# -------------------------------

numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("strategic_correlation_heatmap.png")
plt.show()

print("\nStrategic Analysis Visualizations Created Successfully!")
