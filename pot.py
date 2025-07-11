import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Streamlit page setup
st.set_page_config(page_title="Pot Type Predictor", layout="centered")
st.title("🎯 Next Winning Pot Type Predictor")
st.markdown("📊 Predict whether **High**, **Mid**, or **Low** pot will win in the next round using the last 10 rounds only.")

# Session state for round history
if "rounds" not in st.session_state:
    st.session_state.rounds = []

# Input form for round
st.subheader("➕ Enter Round History")
with st.form("round_form"):
    pot_a = st.number_input("Pot A", min_value=0, step=1)
    pot_b = st.number_input("Pot B", min_value=0, step=1)
    pot_c = st.number_input("Pot C", min_value=0, step=1)
    winner = st.selectbox("Winning Chair", ["A", "B", "C"])
    add_btn = st.form_submit_button("Add Round")

if add_btn:
    st.session_state.rounds.append({"A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner})
    if len(st.session_state.rounds) > 10:
        st.session_state.rounds.pop(0)  # Keep last 10 only
    st.success("✅ Round added!")

# Display round history
if st.session_state.rounds:
    df = pd.DataFrame(st.session_state.rounds)
    st.subheader("🧾 Last 10 Rounds")
    st.dataframe(df, use_container_width=True)

    if len(df) == 10:
        # Convert winner chair to winner pot type
        def get_pot_type(row):
            pots = {"A": row["A"], "B": row["B"], "C": row["C"]}
            sorted_pots = sorted(pots.items(), key=lambda x: x[1])
            rank_map = {
                sorted_pots[0][0]: "Low",
                sorted_pots[1][0]: "Mid",
                sorted_pots[2][0]: "High"
            }
            return rank_map[row["Winner"]]

        df["WinnerPotType"] = df.apply(get_pot_type, axis=1)

        # Create sequence-based training dataset
        X, y = [], []
        for i in range(len(df) - 3):
            seq = df["WinnerPotType"].iloc[i:i+3].tolist()
            target = df["WinnerPotType"].iloc[i+3]
            X.append(seq)
            y.append(target)

        if len(X) > 0:
            # ✅ Fit encoder on full column (X + y)
            all_labels = df["WinnerPotType"].tolist()
            le = LabelEncoder()
            le.fit(all_labels)

            X_enc = [le.transform(x) for x in X]
            y_enc = le.transform(y)

            # Train ML model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_enc, y_enc)

            # Predict using last 3 rounds
            last_seq = df["WinnerPotType"].iloc[-3:].tolist()
            last_seq_enc = le.transform(last_seq).reshape(1, -1)
            pred = model.predict(last_seq_enc)[0]
            pred_label = le.inverse_transform([pred])[0]

            st.subheader("🔮 Prediction")
            st.success(f"🎯 **Next round winner is likely: {pred_label.upper()} pot**")
        else:
            st.warning("⚠️ Not enough sequences to train model. Need 4+ rounds.")
    else:
        st.info("ℹ️ Add exactly 10 rounds to enable prediction.")
else:
    st.info("👈 Start by entering at least one round.")