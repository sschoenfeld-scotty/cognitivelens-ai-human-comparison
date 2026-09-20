import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib, sklearn

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# OneHotEncoder compatibility
major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
ohe_kwargs = dict(handle_unknown="ignore")
if (major, minor) >= (1, 2):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

st.set_page_config(page_title="CognitiveLens — Human-AI Decision Comparison (Auto-binning)", page_icon="🧠", layout="wide")
st.title("🧠 CognitiveLens — Human‑AI Decision Comparison Tool")
st.caption("Auto-binning for numeric fairness attributes is enabled (quartiles).")

# Sidebar: data
st.sidebar.header("1) Data")
up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sample = st.sidebar.selectbox("...or use a sample", ["None","data/sample_decisions.csv"])

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

if up is not None:
    df = pd.read_csv(up)
elif sample != "None":
    df = load_csv(sample)
else:
    st.info("Upload a CSV or pick a sample to begin.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), width='stretch')
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

# Column choices
st.sidebar.header("2) Columns")
target_human = st.sidebar.selectbox(
    "Human label column",
    options=df.columns.tolist(),
    index=(df.columns.get_loc("human_label") if "human_label" in df.columns else 0)
)
target_true = st.sidebar.selectbox(
    "Ground truth (optional)",
    options=["None"] + df.columns.tolist(),
    index=(["None"]+df.columns.tolist()).index("y_true") if "y_true" in df.columns else 0
)
sensitive = st.sidebar.selectbox(
    "Sensitive attribute (fairness)",
    options=["None"] + df.columns.tolist(),
    index=(["None"]+df.columns.tolist()).index("gender") if "gender" in df.columns else 0
)

# Features selection
exclude_cols = {target_human}
if target_true != "None":
    exclude_cols.add(target_true)
feature_candidates = [c for c in df.columns if c not in exclude_cols]
features = st.multiselect("Feature columns (blank = auto-use remaining)",
                          options=feature_candidates, default=feature_candidates)
if not features:
    features = feature_candidates

X = df[features].copy()
y_h = df[target_human].astype(int).copy()
y_true = None if target_true=="None" else df[target_true].astype(int).copy()

# Split num/cat
num_cols = X.select_dtypes(include=["int64","float64","int32","float32","int","float"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Choose model
st.sidebar.header("3) Model")
model_name = st.sidebar.selectbox("Classifier", ["LogisticRegression","RandomForestClassifier"] + (["XGBClassifier"] if HAS_XGB else []))

with st.sidebar.expander("Hyperparameters", expanded=False):
    if model_name == "RandomForestClassifier":
        n_estimators = st.number_input("n_estimators", 50, 1000, 300, 50)
        max_depth = st.number_input("max_depth (0=none)", 0, 50, 0, 1)
    elif model_name == "LogisticRegression":
        C = st.number_input("C", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.number_input("max_iter", 100, 5000, 1000, 100)
    elif model_name == "XGBClassifier" and HAS_XGB:
        xgb_lr = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01)
        xgb_n = st.number_input("n_estimators", 50, 2000, 400, 50)
        xgb_md = st.number_input("max_depth", 1, 20, 6, 1)

# Preprocess + model
numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
categorical_tf = Pipeline(steps=[("ohe", OneHotEncoder(**ohe_kwargs))])
preprocess = ColumnTransformer(
    transformers=[("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)],
    remainder="drop"
)

def make_model():
    if model_name == "LogisticRegression":
        return LogisticRegression(C=C, max_iter=max_iter)
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth), random_state=42)
    if model_name == "XGBClassifier" and HAS_XGB:
        return XGBClassifier(learning_rate=xgb_lr, n_estimators=xgb_n, max_depth=xgb_md, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss")
    raise ValueError("Unsupported model")

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", make_model())])

st.sidebar.header("4) Train/Test")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.25, 0.05)
seed = st.sidebar.number_input("Random state", value=42, step=1)

# Human vs AI comparison target
compare_to = st.radio("What should the AI learn to predict?",
                      ["Ground truth (y_true)", "Human decision (human_label)"],
                      index=0 if y_true is not None else 1)
target_y = y_true if (compare_to=="Ground truth (y_true)" and y_true is not None) else y_h

# Train
if st.button("🚀 Train & Compare"):
    try:
        strat = target_y if target_y.nunique()>1 else None
        split_inputs = [X, target_y, y_h]
        if y_true is not None:
            split_inputs.append(y_true)

        split_data = train_test_split(
            *split_inputs, test_size=test_size, random_state=seed, stratify=strat
        )

        if y_true is not None:
            X_train, X_test, y_train, y_test, h_train, h_test, truth_train, truth_test = split_data
        else:
            X_train, X_test, y_train, y_test, h_train, h_test = split_data
            truth_train = truth_test = None
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # proba if available
        y_prob = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:,1]

        st.success("Training complete.")

        # Metrics block
        c1, c2, c3 = st.columns(3)
        # Agreement with human
        human_agree = accuracy_score(h_test, y_pred)
        with c1: st.metric("Agreement (AI vs Human)", f"{human_agree:.3f}")
        # If ground truth exists, compute accuracy
        if y_true is not None and compare_to=="Ground truth (y_true)":
            acc = accuracy_score(truth_test, y_pred)
            with c2: st.metric("Accuracy (AI vs Truth)", f"{acc:.3f}")
        # Kappa
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(h_test, y_pred)
            with c3: st.metric("Cohen's κ (vs Human)", f"{kappa:.3f}")
        except Exception:
            with c3: st.metric("Cohen's κ (vs Human)", "n/a")

        st.subheader("Confusion Matrix (AI vs Human)")
        cm = confusion_matrix(h_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="AI Pred", y="Human Label", color="Count"))
        st.plotly_chart(fig_cm, config={"responsive": True})

        if y_prob is not None and (y_true is not None or compare_to=="Ground truth (y_true)"):
            # ROC against ground truth only
            if (y_true is not None):
                try:
                    fpr, tpr, _ = roc_curve(truth_test, y_prob)
                    auc = roc_auc_score(truth_test, y_prob)
                    fig = px.area(x=fpr, y=tpr,
                                  labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                  title=f'ROC Curve (AUC={auc:.3f})')
                    st.subheader("ROC Curve (binary)")
                    st.plotly_chart(fig, config={"responsive": True})
                except Exception as e:
                    st.info(f"ROC not available: {e}")

            # Calibration (Brier)
            try:
                brier = brier_score_loss(truth_test, y_prob)
                st.metric("Brier Score (lower is better)", f"{brier:.3f}")
                hist = px.histogram(y_prob, nbins=20, title="Prediction Probabilities")
                st.plotly_chart(hist, config={"responsive": True})
            except Exception:
                pass

        # Fairness analysis by subgroup (AUTO-BINNING for numeric columns)
        st.subheader("Fairness by Subgroup")
        if sensitive != "None" and sensitive in df.columns:
            # use X_test if the sensitive column was part of features; else reference from df
            if sensitive in X_test.columns:
                sens_series = X_test[sensitive]
            else:
                sens_series = df.loc[X_test.index, sensitive]

            # If numeric → bin into quartiles with readable labels
            if np.issubdtype(sens_series.dtype, np.number):
                if sens_series.nunique() < 4:
                    grp_col = sens_series.astype(str)
                else:
                    q = sens_series.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy()
                    q = np.unique(q)
                    if len(q) < 5:
                        q = np.linspace(float(sens_series.min()), float(sens_series.max()), 5)
                    labels = ["Low", "Mid‑Low", "Mid‑High", "High"]
                    grp_col = pd.cut(sens_series, bins=q, labels=labels[:len(q)-1], include_lowest=True, duplicates="drop")
            else:
                grp_col = sens_series.astype(str)

            grp = pd.DataFrame({"group": grp_col})
            grp["human"] = h_test.values
            grp["ai"] = y_pred
            if truth_test is not None:
                grp["truth"] = truth_test.values

            rows = []
            for g, sub in grp.groupby("group", dropna=False):
                if len(sub)==0:
                    continue
                row = {"group": str(g), "n": len(sub),
                       "agree_ai_human": (sub["human"]==sub["ai"]).mean()}
                if truth_test is not None:
                    row["acc_ai_truth"] = (sub["truth"]==sub["ai"]).mean()
                    row["acc_human_truth"] = (sub["truth"]==sub["human"]).mean()
                rows.append(row)
            res = pd.DataFrame(rows).sort_values("n", ascending=False)

            st.dataframe(res, width='stretch')
            if "acc_ai_truth" in res.columns:
                fig_f = px.bar(res, x="group", y=["acc_ai_truth","acc_human_truth"],
                               barmode="group", title="Accuracy by Subgroup")
                st.plotly_chart(fig_f, config={"responsive": True})
            fig_a = px.bar(res, x="group", y="agree_ai_human", title="Agreement (AI vs Human) by Subgroup")
            st.plotly_chart(fig_a, config={"responsive": True})
        else:
            st.info("Pick a sensitive attribute in the sidebar for subgroup analysis.")

        # Downloads
        out = X_test.copy()
        out["human_label"] = h_test
        out["ai_pred"] = y_pred
        if truth_test is not None:
            out["y_true"] = truth_test
        if y_prob is not None:
            out["ai_prob"] = y_prob
        st.download_button("📥 Download evaluation rows (CSV)", data=out.to_csv(index=False), file_name="evaluation_rows.csv", mime="text/csv")

        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        st.download_button("💾 Download trained model (.joblib)", data=buf.getvalue(), file_name="cognitivelens_model.joblib")

    except Exception as e:
        st.error(f"Training failed: {e}")

st.divider()
st.subheader("📝 Survey Mode — collect human judgments")
st.write("Use this to record human labels for new cases and append them to a CSV file.")

with st.form("survey"):
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Income", 0, 300000, 50000, 1000)
    score = st.number_input("Score", -5.0, 5.0, 0.0, 0.1)
    gender = st.selectbox("Gender", ["F","M"])
    region = st.selectbox("Region", ["North","South","East","West"])
    human_label = st.selectbox("Human decision", [0,1], index=0)
    submitted = st.form_submit_button("Append to data/human_judgments.csv")
    if submitted:
        row = pd.DataFrame([{"age":age, "income":income, "score":score, "gender":gender, "region":region, "human_label":human_label}])
        try:
            path = "data/human_judgments.csv"
            try:
                prev = pd.read_csv(path)
                new = pd.concat([prev, row], ignore_index=True)
            except Exception:
                new = row
            new.to_csv(path, index=False)
            st.success("Appended to data/human_judgments.csv")
        except Exception as e:
            st.error(f"Failed to write: {e}")
