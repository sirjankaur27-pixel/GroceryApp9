import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Grocery & Meal Predictor", layout="wide")
st.title("🛒 Smart Grocery & Meal Predictor")
st.caption("Budget-aware meal planning + grocery price forecasting")


# ---------------- Load data (cached so it doesn't reload on every interaction) ----------------
@st.cache_data
def load_data():
    food_data = pd.read_csv("food_dataset.csv").drop_duplicates().reset_index(drop=True)
    grocery_data = pd.read_csv("grocery_prices.csv")
    return food_data, grocery_data


food_data, grocery_data = load_data()
item_cols = [c for c in grocery_data.columns if c not in ("Week", "Date")]

# ---------------- Sidebar: all the inputs live here ----------------
st.sidebar.header("Your targets")
budget = st.sidebar.number_input("Weekly grocery budget (Rs)", min_value=0.0, value=500.0, step=50.0)
calorie_goal = st.sidebar.number_input("Daily calorie goal per person (kcal)", min_value=0, value=2000, step=100)
protein_goal = st.sidebar.number_input("Daily protein goal per person (g)", min_value=0, value=60, step=5)

st.sidebar.header("Personalization")
diet_pref = st.sidebar.selectbox("Dietary preference", ["any", "veg", "non-veg"])
exclude_raw = st.sidebar.text_input("Foods to avoid (comma-separated)", value="")
household_size = st.sidebar.number_input("Household size", min_value=1, value=1, step=1)

st.sidebar.header("Forecast")
weeks_ahead = st.sidebar.slider("Weeks ahead to forecast", min_value=1, max_value=8, value=2)

run = st.sidebar.button("Generate plan", type="primary")

if not run:
    st.info("Set your targets in the sidebar and click **Generate plan**.")
    st.stop()

# ---------------- Filter food catalog ----------------
exclude_list = [x.strip().lower() for x in exclude_raw.split(",") if x.strip()]
available = food_data.copy()
if diet_pref == "veg":
    available = available[available["Type"] == "Veg"]
if exclude_list:
    pattern = "|".join(exclude_list)
    available = available[~available["Food"].str.lower().str.contains(pattern)]

household_calorie_goal = calorie_goal * household_size
household_protein_goal = protein_goal * household_size

if available.empty:
    st.error("No foods left after filtering — loosen your exclusions or diet preference.")
    st.stop()

# ---------------- Greedy meal planner ----------------
picked_idx = []
for cat in available["Category"].unique():
    cat_rows = available[available["Category"] == cat]
    picked_idx.append(cat_rows["Price"].idxmin())
plan_idx = list(dict.fromkeys(picked_idx))

spent = available.loc[plan_idx, "Price"].sum()
got_calories = available.loc[plan_idx, "Calories"].sum()
got_protein = available.loc[plan_idx, "Protein"].sum()

over_budget_warning = None
if spent > budget:
    over_budget_warning = (
        f"Just one item from each food group already costs Rs {spent:.2f}, "
        f"which is over your Rs {budget:.2f} budget. Showing the cheapest possible diverse plan anyway."
    )
else:
    remaining = available.drop(index=plan_idx).copy()
    remaining["Efficiency"] = (
        remaining["Calories"] / max(household_calorie_goal, 1) + remaining["Protein"] / max(household_protein_goal, 1)
    ) / remaining["Price"]
    remaining = remaining.sort_values("Efficiency", ascending=False)

    for idx, row in remaining.iterrows():
        if got_calories >= household_calorie_goal and got_protein >= household_protein_goal:
            break
        if spent + row["Price"] > budget:
            continue
        plan_idx.append(idx)
        spent += row["Price"]
        got_calories += row["Calories"]
        got_protein += row["Protein"]

plan = available.loc[plan_idx].copy()
meal_cost = plan["Price"].sum()
meal_calories = plan["Calories"].sum()
meal_protein = plan["Protein"].sum()
remaining_budget = budget - meal_cost

# ---------------- Price forecasting ----------------
X = grocery_data[["Week"]]
last_week = grocery_data["Week"].max()
future_weeks = pd.DataFrame({"Week": np.arange(last_week + 1, last_week + 1 + weeks_ahead)})

price_models, model_names, future_prices = {}, {}, {"Week": future_weeks["Week"].values}

for item in item_cols:
    y = grocery_data[item]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))
    best_model, best_name = (lr, "LinearRegression") if lr_mae <= rf_mae else (rf, "RandomForest")
    best_model.fit(X, y)
    price_models[item] = best_model
    model_names[item] = best_name
    future_prices[item] = best_model.predict(future_weeks)

future_df = pd.DataFrame(future_prices)
future_df["Total_Cost"] = future_df[item_cols].sum(axis=1)
grocery_data["Total_Cost"] = grocery_data[item_cols].sum(axis=1)

advice_rows = []
for item in item_cols:
    last_price = grocery_data[item].iloc[-1]
    future_price = future_df[item].iloc[-1]
    pct_change = (future_price - last_price) / last_price * 100
    verdict = "Buy now" if pct_change > 2 else ("Wait" if pct_change < -2 else "No urgency")
    advice_rows.append({
        "Item": item, "Current price": round(last_price, 2),
        f"Price in {weeks_ahead}wk": round(future_price, 2),
        "Change %": round(pct_change, 1), "Suggestion": verdict,
    })
advice_df = pd.DataFrame(advice_rows).sort_values("Change %", ascending=False).reset_index(drop=True)

# ================== LAYOUT ==================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Meal plan cost", f"Rs {meal_cost:.2f}")
col2.metric("Remaining budget", f"Rs {remaining_budget:.2f}")
col3.metric("Calories", f"{meal_calories:.0f} / {household_calorie_goal}")
col4.metric("Protein (g)", f"{meal_protein:.1f} / {household_protein_goal}")

if over_budget_warning:
    st.warning(over_budget_warning)

st.subheader("Recommended shopping list")
st.dataframe(plan[["Food", "Category", "Type", "Price", "Calories", "Protein"]], use_container_width=True)

csv_bytes = plan[["Food", "Category", "Type", "Price", "Calories", "Protein"]].to_csv(index=False).encode("utf-8")
st.download_button("Download shopping list (CSV)", csv_bytes, "my_shopping_list.csv", "text/csv")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ax.pie([meal_calories, meal_protein * 4], labels=["Calories (kcal)", "Protein (kcal-equiv)"],
           autopct="%1.1f%%", colors=["#4C72B0", "#55A868"])
    ax.set_title("Nutrition split")
    st.pyplot(fig)
with right:
    fig, ax = plt.subplots()
    ax.bar(plan["Food"], plan["Price"], color="#55A868")
    ax.set_ylabel("Price (Rs)")
    ax.set_title("Cost per item")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    st.pyplot(fig)

st.subheader(f"Price forecast — next {weeks_ahead} week(s)")
st.dataframe(future_df, use_container_width=True)

n_items = len(item_cols)
n_cols = 3
n_rows = -(-n_items // n_cols)  # ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()
for ax, item in zip(axes, item_cols):
    ax.plot(grocery_data["Week"], grocery_data[item], marker="o", label="Actual")
    ax.plot(future_df["Week"], future_df[item], marker="o", linestyle="--", color="red", label="Forecast")
    ax.set_title(f"{item} ({model_names[item]})")
    ax.legend(fontsize=8)
for ax in axes[n_items:]:
    ax.axis("off")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Total weekly cost forecast")
predicted_total = future_df["Total_Cost"].iloc[-1]
st.metric(f"Predicted total spend, week {int(future_df['Week'].iloc[-1])}", f"Rs {predicted_total:.2f}")

fig, ax = plt.subplots()
ax.plot(grocery_data["Week"], grocery_data["Total_Cost"], marker="o", label="Actual weekly cost")
ax.plot(future_df["Week"], future_df["Total_Cost"], marker="o", linestyle="--", color="red", label="Forecast")
ax.set_xlabel("Week")
ax.set_ylabel("Total Cost (Rs)")
ax.set_title("Weekly Grocery Cost Trend & Forecast")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("Buy now or wait?")
st.dataframe(advice_df, use_container_width=True)

st.subheader("Recommendations")
if remaining_budget > 100:
    st.write(f"You have Rs {remaining_budget:.2f} left — consider adding more fruit or vegetables.")
else:
    st.write("Budget is nearly used up — the plan above is tight but on-target.")

if meal_protein < household_protein_goal:
    st.write("Protein is short of goal. Consider adding: Paneer, Eggs (if non-veg), Chickpeas, or Soybean Chunks.")
else:
    st.write("Protein goal met for this plan.")

top_riser = advice_df.iloc[0]
if top_riser["Change %"] > 2:
    st.write(f"**{top_riser['Item']}** is trending up the most (+{top_riser['Change %']}%) — buy it this week.")
else:
    st.write("No item is trending sharply upward — no urgency to stock up on anything specific.")
