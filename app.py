# app.py (patched - Option B)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import altair as alt

st.set_page_config(page_title="Affordability Reality Engine",
                   page_icon="🏫", layout="wide")

#data loading (cached)
@st.cache_data
def load_data():
    affordability_df = pd.read_csv("affordability_raw.csv")
    college_selected_raw = pd.read_csv("college_selected_raw.csv")
    return affordability_df, college_selected_raw

affordability_df, college_selected_raw = load_data()

# helper funcs because this data is so messy
def col_get(row, col_name, default=np.nan):
    return row[col_name] if (hasattr(row, "index") and col_name in row.index) else default

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x

#filter by user preferences, score & rank
def filter_by_state(state, in_out_pref):
    all_states = pd.unique(affordability_df['State Abbreviation']).tolist()
    if in_out_pref == "In-State":
        states = [state]
    elif in_out_pref == "Out-of-State":
        states = [s for s in all_states if s != state]
    else:
        states = all_states
    filtered_df = affordability_df[affordability_df['State Abbreviation'].isin(states)]
    return filtered_df["Unit ID"].tolist()

def filter_by_tuition(tuition_range, in_out_pref, state):
    lower, upper = tuple(tuition_range)
    lower *= 1000
    upper *= 1000
    if in_out_pref == "In-State":
        within_range = college_selected_raw[
            (college_selected_raw["Average In-State Tuition for First-Time, Full-Time Undergraduates"] >= lower) &
            (college_selected_raw["Average In-State Tuition for First-Time, Full-Time Undergraduates"] <= upper)
        ]
    elif in_out_pref == "Out-of-State":
        within_range = college_selected_raw[
            (college_selected_raw["Out-of-State Average Tuition for First-Time, Full-Time Undergraduates"] >= lower) &
            (college_selected_raw["Out-of-State Average Tuition for First-Time, Full-Time Undergraduates"] <= upper)
        ]
    else:
        in_state_ids = affordability_df[affordability_df['State Abbreviation'] == state]["Unit ID"]
        out_of_state_ids = affordability_df[affordability_df['State Abbreviation'] != state]["Unit ID"]
        in_state = college_selected_raw[college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].isin(in_state_ids)]
        out_of_state = college_selected_raw[college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].isin(out_of_state_ids)]
        within_range_in_state = in_state[
            (in_state["Average In-State Tuition for First-Time, Full-Time Undergraduates"] >= lower) &
            (in_state["Average In-State Tuition for First-Time, Full-Time Undergraduates"] <= upper)
        ]
        within_range_out_state = out_of_state[
            (out_of_state["Out-of-State Average Tuition for First-Time, Full-Time Undergraduates"] >= lower) &
            (out_of_state["Out-of-State Average Tuition for First-Time, Full-Time Undergraduates"] <= upper)
        ]
        within_range = pd.concat([within_range_in_state, within_range_out_state], axis=0)
    return within_range["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].tolist()

def filter_by_debt(debt_range):
    lower, upper = tuple(debt_range)
    lower *= 1000
    upper *= 1000
    within_range = college_selected_raw[
        (college_selected_raw["Median Debt for Dependent Students"] >= lower) &
        (college_selected_raw["Median Debt for Dependent Students"] <= upper)
    ]
    return within_range["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].tolist()

def filter_by_minority_serving(require_msi):
    if not require_msi:
        return college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].tolist()
    msi_units = affordability_df[affordability_df['MSI Status'] == 1]["Unit ID"].tolist()
    filtered = college_selected_raw[college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].isin(msi_units)]
    return filtered["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].tolist()

def filter_by_size(size_choice):
    if size_choice == "Small":
        filtered_df = college_selected_raw[college_selected_raw["Number of Undergraduates Enrolled"] <= 5000]
    elif size_choice == "Medium":
        filtered_df = college_selected_raw[(college_selected_raw["Number of Undergraduates Enrolled"] > 5000) &
                                            (college_selected_raw["Number of Undergraduates Enrolled"] <= 15000)]
    else:
        filtered_df = college_selected_raw[college_selected_raw["Number of Undergraduates Enrolled"] > 15000]
    return filtered_df["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].tolist()

def merge_and_normalize(ids):
    if not ids:
        return pd.DataFrame()
    filtered_college = college_selected_raw[college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"].isin(ids)]
    filtered_afford = affordability_df[affordability_df["Unit ID"].isin(ids)][["Unit ID","Institution Name","MSI Status","Average Work Study Award","Affordability Gap (net price minus income earned working 10 hrs at min wage)","State Abbreviation"]]
    merged = filtered_college.merge(filtered_afford, left_on="UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION", right_on="Unit ID", how="inner")
    if merged.empty:
        return merged
    numeric_cols = [
        "Median Earnings of Students Working and Not Enrolled 10 Years After Entry",
        "Median Debt for Dependent Students",
        "Median Debt for Independent Students",
        "Average In-State Tuition for First-Time, Full-Time Undergraduates",
        "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates",
        "Average Amount of Loans Awarded to First-Time, Full-Time Undergraduates",
        "Average Amount of Federal Grant Aid Awarded to First-Time, Full-Time Undergraduates",
        "Average Amount of Institutional Grant Aid Awarded to First-Time, Full-Time Undergraduates",
        "Average Work Study Award",
        "Affordability Gap (net price minus income earned working 10 hrs at min wage)"
    ]
    numeric_cols = [c for c in numeric_cols if c in merged.columns]
    scaler = MinMaxScaler()
    if numeric_cols:
        merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols].fillna(0))
    return merged

def score_and_rank_schools(merged_df, user_weights, column_directions):
    if merged_df.empty:
        return merged_df
    df = merged_df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"]
    df["score"] = 0.0
    for col in numeric_cols:
        weight = user_weights.get(col, 0)
        if weight == 0:
            continue
        vals = df[col].fillna(0)
        if column_directions.get(col) == "lower":
            vals = 1 - vals
        df["score"] += vals * weight
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df.drop_duplicates(subset="Institution Name", keep="first")

#session state defaults
def init_session_state_defaults():
    defaults = {
        "selected_college_id": None,
        "selected_college_name": None,
        "ranked_df": None,
        "merged_df": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state_defaults()

#Input panel!
st.title("🏫 College Finder")

with st.sidebar:
    st.markdown("### Profile & Preferences")
    state = st.selectbox("What state do you live in?", sorted(pd.unique(affordability_df['State Abbreviation'])), key="state")
    in_out_pref = st.radio("I'd like to be...", ["In-State", "Out-of-State", "I don't care"], index=2, key="in_out_pref")
    st.markdown("---")

    with st.expander("Cost Preferences", expanded=True):
        col1, col2 = st.columns([3,1])
        with col1:
            tuition_range = st.slider("Select yearly tuition range (thousands $):", 0, 100, (20, 75), step=1, key="tuition_range")
        with col2:
            tuition_importance = st.slider("Importance", 0, 5, 3, key="tuition_importance", help="How important is tuition when ranking colleges?")
        col1, col2 = st.columns([3,1])
        with col1:
            debt_range = st.slider("Maximum debt you're willing to take (thousands $):", 0, 100, (10, 40), step=1, key="debt_range")
        with col2:
            debt_importance = st.slider("Importance", 0, 5, 3, key="debt_importance", help="How important is minimizing debt for you?")
    st.markdown("---")
    with st.expander("Campus Preferences", expanded=False):
        student_body_size = st.selectbox("Preferred student body size", ["Small", "Medium", "Large"], index=1, key="student_body_size")
        size_importance = st.slider("Importance", 0, 5, 3, key="size_importance")
        msi_required = st.checkbox("Require Minority-Serving Institution (MSI)?", key="msi_required")
        msi_importance = st.slider("MSI importance in ranking", 0, 5, 1, key="msi_importance")
    st.markdown("---")

    st.markdown("#### Weights preview (you can tweak importance sliders)")
    user_weights = {
        "Median Debt for Dependent Students": st.session_state.get("debt_importance", 3),
        "Median Debt for Independent Students": st.session_state.get("debt_importance", 3),
        "Average In-State Tuition for First-Time, Full-Time Undergraduates": st.session_state.get("tuition_importance", 3),
        "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates": st.session_state.get("tuition_importance", 3),
        "Average Amount of Loans Awarded to First-Time, Full-Time Undergraduates": st.session_state.get("tuition_importance", 3),
        "Average Amount of Federal Grant Aid Awarded to First-Time, Full-Time Undergraduates": st.session_state.get("tuition_importance", 3),
        "Average Amount of Institutional Grant Aid Awarded to First-Time, Full-Time Undergraduates": st.session_state.get("tuition_importance", 3),
        "Average Work Study Award": st.session_state.get("tuition_importance", 3),
        "Affordability Gap (net price minus income earned working 10 hrs at min wage)": st.session_state.get("tuition_importance", 3),
        "MSI Status": st.session_state.get("msi_importance", 1),
        "Median Earnings of Students Working and Not Enrolled 10 Years After Entry": 3
    }
    st.markdown("**Tip:** importance sliders range 0 (ignored) to 5 (crucial).")
    st.markdown("---")

    
    def compute_recommendations_callback():
        # read values from session_state
        state_val = st.session_state["state"]
        in_out_val = st.session_state["in_out_pref"]
        tuition_val = st.session_state["tuition_range"]
        debt_val = st.session_state["debt_range"]
        msi_val = st.session_state["msi_required"]
        size_val = st.session_state["student_body_size"]

        # filters
        state_ids = filter_by_state(state_val, in_out_val)
        tuition_ids = filter_by_tuition(tuition_val, in_out_val, state_val)
        debt_ids = filter_by_debt(debt_val)
        msi_ids = filter_by_minority_serving(msi_val)
        size_ids = filter_by_size(size_val)

        #find intersection of all prefs
        found_ids = list(set(state_ids) & set(tuition_ids) & set(debt_ids) & set(msi_ids) & set(size_ids))
        if not found_ids:
            st.session_state.ranked_df = pd.DataFrame()
            st.session_state.merged_df = pd.DataFrame()

            # show warning after rerun
            st.session_state._last_warning = "No colleges match your filters."
            return

        merged = merge_and_normalize(found_ids)
        if merged.empty:
            st.session_state.ranked_df = pd.DataFrame()
            st.session_state.merged_df = merged
            st.session_state._last_warning = "After merging datasets, no colleges had the required fields."
            return

        # dynamic column directions: default to higher unless known lower
        column_directions = {}
        lower_is_better = {
            "Median Debt for Dependent Students",
            "Median Debt for Independent Students",
            "Average In-State Tuition for First-Time, Full-Time Undergraduates",
            "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates",
            "Affordability Gap (net price minus income earned working 10 hrs at min wage)"
        }
        for c in merged.columns:
            column_directions[c] = "lower" if c in lower_is_better else "higher"

        ranked = score_and_rank_schools(merged, user_weights, column_directions)
        st.session_state.ranked_df = ranked
        st.session_state.merged_df = merged
        st.session_state._last_warning = None

    st.button("GO! Show Recommendations", type="primary", on_click=compute_recommendations_callback, key="go_button")


# Output: show warnings or ranked results if they exist in session_state

if st.session_state.get("_last_warning"):
    st.warning(st.session_state._last_warning)

ranked_df = st.session_state.get("ranked_df", None)
merged_df = st.session_state.get("merged_df", None)

if ranked_df is None or ranked_df.empty:
    st.info("No recommendations yet — set filters on the left and click GO!")
else:
    top_n = 9
    top = ranked_df.head(top_n).copy()
    st.subheader(f"Top {min(top_n, len(top))} Recommendations")
    st.markdown("Click a college card's View details button to open its full detail view on the right.")

    cols = st.columns(3)
    # callback to set selected college (safe, no inline mutation during rerun)
    def select_college_callback(unit_id, name):
        st.session_state.selected_college_id = unit_id
        st.session_state.selected_college_name = name

    for i, row in top.reset_index().iterrows():
        c_idx = i % 3
        with cols[c_idx]:
            name = col_get(row, "Institution Name", "Unknown")
            score = round(col_get(row, "score", 0) * 100, 1)
            st.markdown(f"### {name}")
            st.caption(f"Recommendation score: **{score}**")
            sn1, sn2 = st.columns(2)
            earnings = col_get(row, "Median Earnings of Students Working and Not Enrolled 10 Years After Entry", np.nan)
            dep_debt = col_get(row, "Median Debt for Dependent Students", np.nan)
            sn1.metric("Median Earnings (10y)", f"${safe_int(earnings):,}" if not pd.isna(earnings) else "N/A")
            sn2.metric("Median Debt (dependent)", f"${safe_int(dep_debt):,}" if not pd.isna(dep_debt) else "N/A")
            # tuition mini-chart
            tuition_vals = []
            tuition_labels = []
            if "Average In-State Tuition for First-Time, Full-Time Undergraduates" in merged_df.columns:
                tuition_vals.append(col_get(row, "Average In-State Tuition for First-Time, Full-Time Undergraduates", 0))
                tuition_labels.append("In-State")
            if "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates" in merged_df.columns:
                tuition_vals.append(col_get(row, "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates", 0))
                tuition_labels.append("Out-of-State")
            if tuition_vals:
                tdf = pd.DataFrame({"Type": tuition_labels, "Cost": tuition_vals})
                chart = alt.Chart(tdf).mark_bar(size=12).encode(
                    x=alt.X('Cost:Q', title='Cost ($)'),
                    y=alt.Y('Type:N', sort='-x', title=None),
                ).properties(height=80)
                st.altair_chart(chart, use_container_width=True)

            # Use on_click callback with args instead of inline st.button in if-statement
            unit_id = int(col_get(row, "UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION", -1))
            btn_key = f"view_{unit_id}"
            st.button("View details", key=btn_key,
                      on_click=select_college_callback,
                      args=(unit_id, name))

    st.markdown("---")
    st.header("Selected College — Details")
    selected_name = st.session_state.get("selected_college_name", None)
    selected_id = st.session_state.get("selected_college_id", None)
    if selected_name and selected_id:
        aff_row = affordability_df[affordability_df["Unit ID"] == selected_id]
        sel_row = college_selected_raw[college_selected_raw["UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION"] == selected_id]
        if not aff_row.empty and not sel_row.empty:
            aff_row = aff_row.iloc[0]
            sel_row = sel_row.iloc[0]

            st.subheader(selected_name)
            st.metric("State", aff_row.get("State Abbreviation", "N/A"))
            st.metric("Undergraduate Enrollment", f"{safe_int(col_get(sel_row, 'Number of Undergraduates Enrolled', 0)):,}")
            st.metric("MSI Status", "Yes" if col_get(aff_row, "MSI Status", 0) == 1 else "No")

            tabs = st.tabs(["Tuition & Cost", "Debt & Earnings", "Demographics"])
            with tabs[0]:
                st.write("#### Tuition breakdown")
                tuition_df = pd.DataFrame({
                    "Category": ["In-State", "Out-of-State"],
                    "Cost": [
                        col_get(sel_row, "Average In-State Tuition for First-Time, Full-Time Undergraduates", np.nan),
                        col_get(sel_row, "Out-of-State Average Tuition for First-Time, Full-Time Undergraduates", np.nan)
                    ]
                }).dropna()
                if not tuition_df.empty:
                    c = alt.Chart(tuition_df).mark_bar().encode(
                        x=alt.X("Cost:Q", title="Annual Cost ($)"),
                        y=alt.Y("Category:N", sort='-x', title=None)
                    )
                    st.altair_chart(c, use_container_width=True)
                grant = col_get(sel_row, 'Average Amount of Institutional Grant Aid Awarded to First-Time, Full-Time Undergraduates', np.nan)
                st.write("**Average Institutional Grant Aid**: ",
                         f"${safe_int(grant):,}" if not pd.isna(grant) else "N/A")

            with tabs[1]:
                st.write("#### Debt & Earnings")
                earnings = col_get(sel_row, "Median Earnings of Students Working and Not Enrolled 10 Years After Entry", np.nan)
                dependent_debt = col_get(sel_row, "Median Debt for Dependent Students", np.nan)
                independent_debt = col_get(sel_row, "Median Debt for Independent Students", np.nan)
                st.metric("Median Earnings (10y)", f"${safe_int(earnings):,}" if not pd.isna(earnings) else "N/A")
                st.metric("Median Debt (Dependent)", f"${safe_int(dependent_debt):,}" if not pd.isna(dependent_debt) else "N/A")
                st.metric("Median Debt (Independent)", f"${safe_int(independent_debt):,}" if not pd.isna(independent_debt) else "N/A")

                if not pd.isna(earnings) and earnings > 0 and not pd.isna(dependent_debt):
                    d_to_e = dependent_debt / earnings
                    st.write(f"**Debt-to-Earnings ratio (dependent debt / earnings)**: {d_to_e:.2f}")
                    gauge_df = pd.DataFrame({"metric": ["Ratio"], "value": [d_to_e]})
                    g = alt.Chart(gauge_df).mark_bar().encode(
                        x=alt.X('value:Q', scale=alt.Scale(domain=[0, max(5, d_to_e + 1)]), title="Debt-to-earnings"),
                        y=alt.Y('metric:N', title=None)
                    ).properties(height=50)
                    st.altair_chart(g, use_container_width=True)

            with tabs[2]:
                st.write("#### Student Race/Ethnicity (percent)")
                race_map = {
                    "American Indian or Alaska Native": "Percent of American Indian or Alaska Native Undergraduates",
                    "Two or More Races": "Percent of Two or More Races Undergraduates",
                    "Asian": "Percent of Asian Undergraduates",
                    "Black": "Percent of Black or African American Undergraduates",
                    "Latino": "Percent of Latino Undergraduates",
                    "Native Hawaiian or Other Pacific Islander": "Percent of Native Hawaiian or Other Pacific Islander Undergraduates",
                    "White": "Percent of White Undergraduates",
                    "Unknown": "Percent of Undergraduates Race-Ethnicity Unknown"
                }
                rows = []
                for label, col in race_map.items():
                    if col in sel_row.index:
                        val = col_get(sel_row, col, np.nan)
                        if not pd.isna(val):
                            rows.append({"race": label, "pct": val})
                rdf = pd.DataFrame(rows)
                if not rdf.empty:
                    pie = alt.Chart(rdf).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="pct", type="quantitative"),
                        color=alt.Color(field="race", type="nominal"),
                        tooltip=["race", "pct"]
                    )
                    st.altair_chart(pie, use_container_width=True)
                else:
                    st.info("No race/ethnicity percentage data available for this institution.")
        else:
            st.error("No detailed statistics found for this institution.")
    else:
        st.info("Select a college card to see details here.")

st.markdown("---")
with st.container():
    st.caption("Tip: inputs have been made persistent. Use GO to refresh recommendations; click View details to pin a college without losing your filters.")
