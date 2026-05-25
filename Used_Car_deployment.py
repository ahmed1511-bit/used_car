import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .model-card {
        background: linear-gradient(135deg, #1a1f35 0%, #242840 100%);
        border-radius: 12px;
        padding: 13px 18px;
        margin-bottom: 8px;
        border-left: 4px solid #5c85ff;
        color: #d0d6f0;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        transition: border-color 0.2s;
    }
    .model-card:hover {
        border-left-color: #ff6b6b;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-left: 8px;
        letter-spacing: 0.4px;
    }
    .badge-best  { background: #2d6a4f; color: #95d5b2; border: 1px solid #52b788; }
    .badge-poly  { background: #1b3a6b; color: #90b4f7; border: 1px solid #4a7fd4; }
    .badge-base  { background: #2e2e3a; color: #a0a0b8; border: 1px solid #55556a; }
</style>
""", unsafe_allow_html=True)

# ── Model Registry ────────────────────────────────────────────────────────
# Each entry: (display_label, file_key, needs_prep, needs_poly)
#   needs_prep = False  → model is a full sklearn Pipeline (includes preprocessor)
#   needs_prep = True   → standalone model; call prep.transform() first
#   needs_poly = True   → apply PolynomialFeatures(degree=2) after prep

MODEL_REGISTRY = [
    {
        "label":       "1️⃣  Linear Regression",
        "badge":       ("base", "Baseline"),
        "file":        "linear regression.joblib",
        "needs_prep":  False,
        "needs_poly":  False,
    },
    {
        "label":       "2️⃣  Poly Linear Regression",
        "badge":       ("poly", "Polynomial"),
        "file":        "poly linear regression.joblib",
        "needs_prep":  False,
        "needs_poly":  False,   # poly is inside the pipeline
    },
    {
        "label":       "3️⃣  XGBoost Default",
        "badge":       ("base", "Baseline"),
        "file":        "XGBoost Deafult.joblib",
        "needs_prep":  False,
        "needs_poly":  False,
    },
    {
        "label":       "4️⃣  Poly XGBoost Default",
        "badge":       ("poly", "Polynomial"),
        "file":        "Poly XGBoost Deafult.joblib",
        "needs_prep":  False,
        "needs_poly":  False,   # poly is inside the pipeline
    },
    {
        "label":       "5️⃣  XGBoost optimized (Early Stopping) ⭐",
        "badge":       ("best", "Best Model"),
        "file":        "XGBoost optmizing.joblib",
        "needs_prep":  True,
        "needs_poly":  False,
    },
    {
        "label":       "6️⃣  Poly XGBoost optimized",
        "badge":       ("poly", "Polynomial"),
        "file":        "Poly XGBoost optmizing.joblib",
        "needs_prep":  True,
        "needs_poly":  True,
    },
    {
        "label":       "7️⃣  LightGBM",
        "badge":       ("base", "Fast"),
        "file":        "LightGBM Model.joblib",
        "needs_prep":  True,
        "needs_poly":  False,
    },
    {
        "label":       "8️⃣  Poly LightGBM",
        "badge":       ("poly", "Polynomial"),
        "file":        "Poly LightGBM Model.joblib",
        "needs_prep":  True,
        "needs_poly":  True,
    },
]

MODEL_LABELS = [m["label"] for m in MODEL_REGISTRY]

# ── Load all assets ───────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    prep = joblib.load("preprocessor.joblib")
    models = {}
    errors = {}
    for entry in MODEL_REGISTRY:
        try:
            models[entry["label"]] = joblib.load(entry["file"])
        except Exception as e:
            errors[entry["label"]] = str(e)
    return prep, models, errors

prep, loaded_models, load_errors = load_all()

if load_errors:
    st.warning("⚠️ بعض الموديلات مش موجودة:")
    for name, err in load_errors.items():
        st.caption(f"• {name}: `{err}`")

# ── Header ────────────────────────────────────────────────────────────────
st.title("🚗 Used Car Price Predictor")
st.markdown("ادخل بيانات السيارة وهنحسبلك السعر المتوقع بأي موديل من الـ 8 موديلات")
st.divider()

# ── Model Selector ────────────────────────────────────────────────────────
st.subheader("🤖 اختار الموديل")

badge_html = {
    "best":  '<span class="badge badge-best">⭐ Best</span>',
    "poly":  '<span class="badge badge-poly">Polynomial</span>',
    "base":  '<span class="badge badge-base">Baseline</span>',
}

# Show info cards
for m in MODEL_REGISTRY:
    btype, blabel = m["badge"]
    st.markdown(
        f'<div class="model-card">'
        f'{m["label"]} '
        f'<span class="badge badge-{btype}">{blabel}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

model_choice = st.selectbox(
    "اختار الموديل اللي عايز تستخدمه:",
    options=MODEL_LABELS,
    index=4,   # XGBoost Tuned default
)
st.divider()

# ── Car Details Inputs ────────────────────────────────────────────────────
st.subheader("📋 Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Year", min_value=2000, max_value=2022, value=2018, step=1
    )
    manufacturer = st.selectbox(
        "Manufacturer",
        sorted(["ford","chevrolet","toyota","honda","nissan","jeep","ram",
                "gmc","bmw","mercedes-benz","dodge","hyundai","kia",
                "volkswagen","subaru","audi","lexus","cadillac","buick",
                "lincoln","acura","infiniti","volvo","chrysler","pontiac",
                "saturn","mitsubishi","mazda","mini","land rover",
                "jaguar","rover","tesla","other"])
    )
    model_name = st.text_input("Model", value="f-150")
    condition  = st.selectbox(
        "Condition",
        ["new","like new","excellent","good","fair","salvage"],
        index=3
    )
    cylinders = st.selectbox(
        "Cylinders",
        [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        index=4
    )
    fuel = st.selectbox("Fuel Type", ["gas","diesel","hybrid","electric"])

with col2:
    odometer = st.number_input(
        "Odometer (miles)",
        min_value=1_000, max_value=2_000_000, value=75_000, step=1_000
    )
    title_status = st.selectbox(
        "Title Status",
        ["clean","rebuilt","lien","salvage","missing","parts only"]
    )
    transmission = st.selectbox("Transmission", ["automatic","manual"])
    drive        = st.selectbox("Drive", ["4wd","fwd","rwd"])
    car_type     = st.selectbox(
        "Type",
        sorted(["pickup","SUV","sedan","truck","coupe","hatchback",
                "wagon","van","minivan","convertible","offroad","bus","other"])
    )
    state = st.selectbox(
        "State",
        sorted(["al","ak","az","ar","ca","co","ct","de","fl","ga",
                "hi","id","il","in","ia","ks","ky","la","me","md",
                "ma","mi","mn","ms","mo","mt","ne","nv","nh","nj",
                "nm","ny","nc","nd","oh","ok","or","pa","ri","sc",
                "sd","tn","tx","ut","vt","va","wa","wv","wi","wy","dc"]),
        index=4
    )

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    predict_one = st.button("🔮 Predict — الموديل المختار", type="primary", use_container_width=True)

with col_btn2:
    predict_all = st.button("🏆 كل الموديلات", use_container_width=True)

input_df = pd.DataFrame([{
    "year":         year,
    "manufacturer": manufacturer,
    "model":        model_name,
    "condition":    condition,
    "cylinders":    cylinders,
    "fuel":         fuel,
    "odometer":     odometer,
    "title_status": title_status,
    "transmission": transmission,
    "drive":        drive,
    "type":         car_type,
    "state":        state
}])


def run_model(entry, model):
    """Run inference for a single model entry and return price (float)."""
    needs_prep = entry["needs_prep"]
    needs_poly = entry["needs_poly"]

    if needs_prep:
        X_t = prep.transform(input_df)
        if needs_poly:
            poly_feats = PolynomialFeatures(degree=2, include_bias=False)
            X_t = poly_feats.fit_transform(X_t)
        log_price = model.predict(X_t)[0]
    else:
        log_price = model.predict(input_df)[0]

    return float(np.exp(log_price))


# ── Single Model Prediction ───────────────────────────────────────────────
if predict_one:
    entry = next(m for m in MODEL_REGISTRY if m["label"] == model_choice)
    if model_choice in load_errors:
        st.error(f"❌ الموديل ده مش محمل: {load_errors[model_choice]}")
    else:
        try:
            model = loaded_models[model_choice]
            price = run_model(entry, model)
            st.success(f"### 💰 Predicted Price:  ${price:,.0f}")
            st.info(
                f"**{year} {manufacturer.title()} {model_name.title()}**  \n"
                f"Condition: `{condition}` &nbsp;|&nbsp; "
                f"Odometer: `{odometer:,} miles` &nbsp;|&nbsp; "
                f"Fuel: `{fuel}` &nbsp;|&nbsp; "
                f"Drive: `{drive.upper()}`"
            )
            st.caption(
                f"📊 Estimated range: ${price*0.9:,.0f} – ${price*1.1:,.0f}  (±10%)  "
                f"| Model: {model_choice}"
            )
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")


# ── All Models Comparison ─────────────────────────────────────────────────
if predict_all:
    st.subheader("📊 مقارنة كل الموديلات")

    results = []
    progress = st.progress(0, text="⏳ بيتحسب...")
    total = len(MODEL_REGISTRY)

    for i, entry in enumerate(MODEL_REGISTRY):
        label = entry["label"]
        progress.progress((i + 1) / total, text=f"⏳ {label}")
        if label in load_errors:
            results.append({"Model": label, "Predicted Price": None, "Status": "❌ Not loaded"})
            continue
        try:
            model = loaded_models[label]
            price = run_model(entry, model)
            results.append({
                "Model": label,
                "Predicted Price ($)": f"${price:,.0f}",
                "Low (-10%)": f"${price*0.9:,.0f}",
                "High (+10%)": f"${price*1.1:,.0f}",
            })
        except Exception as e:
            results.append({"Model": label, "Predicted Price ($)": f"Error: {e}", "Low (-10%)": "-", "High (+10%)": "-"})

    progress.empty()

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # highlight best (XGBoost Tuned usually)
    valid = [r for r in results if "Predicted Price ($)" in r and r["Predicted Price ($)"].startswith("$")]
    if valid:
        st.caption("💡 الموديلات الـ Poly والـ Tuned بتديك أدق نتيجة عادةً")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "8 Models: Linear Regression · Poly LR · XGBoost Default · Poly XGBoost Default · "
    "XGBoost Tuned · Poly XGBoost Tuned · LightGBM · Poly LightGBM  |  "
    "Trained on vehicles.csv | log(price) target | used_cars_pipeline.ipynb"
)
