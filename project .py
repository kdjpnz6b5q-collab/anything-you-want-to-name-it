files = {

    "netherlands": [
        "~/Downloads/listings.csv",
        "~/Downloads/listings-2.csv",
        "~/Downloads/listings-3.csv",
        "~/Downloads/listings-4.csv"
    ],
    "spain": [
        "~/Downloads/listings-5.csv",
        "~/Downloads/listings-6.csv",
        "~/Downloads/listings-7.csv",
        "~/Downloads/listings-8.csv"
    ],
    "ireland": [
        "~/Downloads/listings-9.csv",
        "~/Downloads/listings-10.csv",
        "~/Downloads/listings-11.csv",
        "~/Downloads/listings-12.csv"
    ],
    "france": [
        "~/Downloads/listings-13.csv",
        "~/Downloads/listings-14.csv",
        "~/Downloads/listings-15.csv",
        "~/Downloads/listings-16.csv"
    ]
}
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

# the snapshot dates coming from airnbnb, we have 4 scraping dates for each location
snapshots = [
    "2025-09-11",
    "2025-06-09",
    "2025-03-02",
    "2024-12-07"
]

# we want to cmbine the four cities, with teh four different scrapping times, and keep it long format
def load_city(city_name, file_list, snapshot_list):
    frames = []
    for file_path, snap in zip(file_list, snapshot_list):
        df = pd.read_csv(file_path)
        df["snapshot"] = snap
        df["city"] = city_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

all_cities = []

for city, file_list in files.items():
    print("Loading:", city)
    city_df = load_city(city, file_list, snapshots)
    all_cities.append(city_df)

airbnb = pd.concat(all_cities, ignore_index=True)

# Time to solter them by snapshot date first by id number
airbnb["snapshot"] = pd.to_datetime(airbnb["snapshot"])
airbnb = airbnb.sort_values(["city", "id", "snapshot"])

# we have 81 columns, which many of them we dont need. time to drop some
cols_to_drop = [
    "listing_url", "description", "neighborhood_overview", 
    "picture_url", "host_url", "host_about", "host_thumbnail_url", 
    "host_picture_url", "scrape_id", "last_scraped", "source", "host_verifications", "host_has_profile_pic", "host_identity_verified",
    "host_location", "host_name", "host_neighbourhood", "calendar_updated", "calendar_last_scraped",
    "availability_30", "availability_60", "availability_90", "availability_365", "availability_eoy", "minimum_nights", "maximum_nights", 
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", "license", "instant_bookable",
    "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms"
]
airbnb = airbnb.drop(columns=cols_to_drop, errors="ignore")

# for the price number to work we need to clean the data. 
airbnb["price"] = (
    airbnb["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("", None)
    .astype(float)
)

# for host response rate, some values use % and some dont, some entires are emptie and some are a string
airbnb["host_response_rate"] = (
    airbnb["host_response_rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace("nan", "", regex=False)
    .str.strip()
    .replace(["", "N/A"], None)
    .astype(float)
)

model_df["host_acceptance_rate"] = (
    model_df["host_acceptance_rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace("nan", "", regex=False)
    .str.strip()
    .replace(["", "N/A"], None)
    .astype(float)
)


# all the revies scores need to be cleaned up by:
review_cols = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]
for col in review_cols:
    airbnb[col] = pd.to_numeric(airbnb[col], errors="coerce")

# snapshot date to time
airbnb["snapshot"] = pd.to_datetime(airbnb["snapshot"])


# going to process whehere i ran (airbnb.isnull().mean().sort_values(ascending=False) * 100).round(2)
# i figured out data a lot of things are wrong with the data, dropping columns neighbourhood_group_cleansed,
# estimated_revenue_l365d  and neighbourhood   due to the fact we have more than 50% missing data. 

# for price we have missing data of 43%
airbnb.groupby("city")["price"].apply(lambda x: x.isna().mean() * 100)
#conclusion, a lot of data missing for france and netherlands. France regulations dont allow in some circustamces not to share prices
# that why mnore and more price are not avaialbe.
# overall a lot of prices are missing due to the season renting, people might only rent in the summer. 



price_counts = airbnb.groupby(["city", "id"])["price"].apply(lambda x: x.notna().sum())
full_cycle_ids = price_counts[price_counts == 4].index
n_full_cycle = len(full_cycle_ids)
print("Listings with all 4 snapshots:", n_full_cycle)
percentage = n_full_cycle / len(price_counts) * 100
print(f"Percentage of total: {percentage:.2f}%")
#conclusion: only 10.13 of properties have full price data for four concesitive periode, but we need at least two concetive periods 

valid_ids = price_counts[price_counts >= 2].index
airbnb_2plus = airbnb[airbnb.set_index(["city", "id"]).index.isin(valid_ids)]
airbnb_2plus = airbnb_2plus.sort_values(["city", "id", "snapshot"])
airbnb_2plus["price_prev"] = airbnb_2plus.groupby(["city", "id"])["price"].shift(1)
airbnb_2plus["price_change"] = airbnb_2plus["price"] - airbnb_2plus["price_prev"]
airbnb_2plus["price_up"] = (airbnb_2plus["price_change"] > 0).astype(int)
model_df = airbnb_2plus.dropna(subset=["price_prev", "price"])




######
# we only want to built this linear regression for Amsterdam, on snapshot 09-11
nl = airbnb[airbnb["city"] == "netherlands"].copy()
nl_latest = nl[nl["snapshot"] == pd.to_datetime("2025-09-11")].copy()

# price, remove signs and komma
nl_latest["price"] = (
    nl_latest["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace("", None)
    .astype(float)
)

# ilter prices under 1000 dollar a night, i didnt limit this at first but models where awfull
nl_clean = nl_latest[nl_latest["price"] <= 1000].copy()
nl_clean = nl_clean.dropna(subset=["price"])

# use some flags for superhost or not (1/0)
nl_clean["host_is_superhost"] = nl_clean["host_is_superhost"].map({"t": 1, "f": 0})

# Clean host acceptance rate
nl_clean["host_acceptance_rate"] = (
    nl_clean["host_acceptance_rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace("nan", "", regex=False)
    .str.strip()
    .replace(["", "N/A"], None)
    .astype(float)
)

# Clean the review scores
review_cols = [
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location"
]

for col in review_cols:
    nl_clean[col] = pd.to_numeric(nl_clean[col], errors="coerce")


nl_clean["log_price"] = np.log(nl_clean["price"])
nl_encoded = pd.get_dummies(nl_clean, columns=["property_type"], drop_first=True)
base_features = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate"
]

property_dummies = [c for c in nl_encoded.columns if c.startswith("property_type_")]

features = base_features + property_dummies


X = nl_encoded[features]
y = nl_encoded["log_price"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lr", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)

log_preds = model.predict(X_test)
pred_price = np.exp(log_preds)
actual_price = np.exp(y_test)

mse = mean_squared_error(actual_price, pred_price)


# lets check the RMSE , 120 dollar difference, not bad for first model
print("RMSE:", rmse)


# okay i want to have more specifc model, becasue location is very important
base_features = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "latitude",    # NEW
    "longitude"    # NEW
]

property_dummies = [c for c in nl_encoded.columns if c.startswith("property_type_")]
features = base_features + property_dummies

X = nl_encoded[features]
y = nl_encoded["log_price"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lr", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model.fit(X_train, y_train)

log_preds = model.predict(X_test)
pred_price = np.exp(log_preds)
actual_price = np.exp(y_test)

mse = mean_squared_error(actual_price, pred_price)
print("RMSE:", rmse)
# great test, but does not change anything


# time for nonlinear models
X = nl_encoded[features]
y = nl_encoded["price"]      # NOTE: RandomForest uses raw price, NOT log-price

from sklearn.model_selection import train_test_split
#train versus test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

from sklearn.ensemble import RandomForestRegressor
#fit random forest regresssion
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
import numpy as np

preds = rf.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("RMSE:", rmse)



pip install shap
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="dot", show=True)
