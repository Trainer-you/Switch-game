import streamlit as st
import pandas as pd

st.set_page_config(page_title="Teen Patti Smart Predictor", layout="centered")
st.title("🃏 Teen Patti Chair Predictor (Dual Mode)")
st.markdown("Switch between **Pattern Mode** and **Pot Mode** to adapt to game strategy.")

if "rounds" not in st.session_state:
    st.session_state.rounds = []

# Mode selection
mode = st.radio("🎮 Select Prediction Mode:", ["Pattern Mode", "Pot Mode"])

# Input form
chairs = ["A", "B", "C"]
with st.form("round_input"):
    st.subheader("➕ Add Round Data")
    pot_a = st.number_input("Pot A (in thousands)", min_value=0, step=1)
    pot_b = st.number_input("Pot B (in thousands)", min_value=0, step=1)
    pot_c = st.number_input("Pot C (in thousands)", min_value=0, step=1)
    winner = st.selectbox("Winning Chair", chairs)
    submit = st.form_submit_button("Add Round")

if submit:
    st.session_state.rounds.append({"Round": len(st.session_state.rounds)+1, "A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner})
    st.success(f"✅ Round {len(st.session_state.rounds)} added")

# Show history
if st.session_state.rounds:
    df = pd.DataFrame(st.session_state.rounds)
    st.subheader("📜 Match History (Last 10)")
    st.dataframe(df.tail(10), use_container_width=True)

    def pattern_mode_predictor(df):
        recent = df.tail(10)
        scores = {c: 0 for c in chairs}
        reasons = {c: [] for c in chairs}
        last_3 = recent["Winner"].tolist()[-3:] if len(recent) >= 3 else []

        for chair in chairs:
            if last_3 == ['A', 'B', 'C'] and chair == 'A':
                scores[chair] += 4
                reasons[chair].append("+4 for A→B→C→A pattern")
            elif last_3 == ['B', 'C', 'A'] and chair == 'B':
                scores[chair] += 4
                reasons[chair].append("+4 for B→C→A→B pattern")
            elif last_3 == ['C', 'A', 'B'] and chair == 'C':
                scores[chair] += 4
                reasons[chair].append("+4 for C→A→B→C pattern")
            scores[chair] += 5 - recent["Winner"].tolist().count(chair)
            reasons[chair].append("+? for fewer recent wins")
        return scores, reasons

    def pot_mode_predictor(df):
        recent = df.tail(10)
        last = recent.iloc[-1][["A", "B", "C"]]
        scores = {c: 0 for c in chairs}
        reasons = {c: [] for c in chairs}

        for chair in chairs:
            chair_pot = last[chair]
            if chair_pot == min(last):
                scores[chair] += 2
                reasons[chair].append("+2 for lowest pot last round")
            elif chair_pot == sorted(last)[1]:
                scores[chair] += 1
                reasons[chair].append("+1 for medium pot")
            scores[chair] += 5 - recent["Winner"].tolist().count(chair)
            reasons[chair].append("+? for fewer recent wins")
        return scores, reasons

    def suggest_mode(df):
        recent = df.tail(3)
        last_pots = df.iloc[-1][["A", "B", "C"]]
        winner_pattern = recent["Winner"].tolist()
        pattern_match = winner_pattern in [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]
        pot_gap = max(last_pots) - min(last_pots) > 300

        if pattern_match and not pot_gap:
            return "Pattern Mode"
        elif pot_gap and not pattern_match:
            return "Pot Mode"
        else:
            return "Pot Mode (default)"

    suggested = suggest_mode(df)
    st.markdown(f"### 🧠 Suggested Mode: **{suggested}**")

    # Run prediction
    if mode == "Pattern Mode":
        scores, reasons = pattern_mode_predictor(df)
    else:
        scores, reasons = pot_mode_predictor(df)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    st.subheader(f"🔮 Prediction for Next Round ({mode})")
    for idx, (chair, score) in enumerate(sorted_scores[:2], start=1):
        st.markdown(f"**{idx}. Chair {chair}** — Score: `{score}`")
        with st.expander(f"📌 Reasoning for Chair {chair}"):
            for reason in reasons[chair]:
                st.markdown(f"- {reason}")
else:
    st.info("Please enter at least 1 round to start predictions.")
