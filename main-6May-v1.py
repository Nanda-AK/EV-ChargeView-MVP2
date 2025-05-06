# EV Review Streamlit App with GPT-4o and 10 Predefined Analysis Functions

import streamlit as st
import pandas as pd
import json
import openai
from datetime import datetime
from pandasai import SmartDataframe
#from pandasai.llm.openai import OpenAI
#from openai import OpenAI

from pandasai.llm.openai import OpenAI as PandasAIOpenAI
from openai import OpenAI  # This is the official OpenAI SDK client

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="EV Review Insights", layout="wide")

# --- CONFIG ---
openai.api_key = st.secrets["OpenAI_API_KEY"]  # Store key in Streamlit secrets
#llm = OpenAI(api_token=openai.api_key, model="gpt-3.5-turbo")
llm = PandasAIOpenAI(api_token=openai.api_key, model="gpt-3.5-turbo")
#llm = OpenAI(api_token=openai.api_key, model="gpt-4o-mini")

# --- LOAD JSON DATA ---
@st.cache_data
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

raw_df = load_data("cleaned_ev_data.json")

# --- FILTER INCOMPLETE RECORDS (no drops, just filter) ---
raw_df = raw_df[
    raw_df['EV Vendor'].notna() &
    raw_df['address'].notna() &
    raw_df['reviewsCount'].notna() &
    raw_df['totalScore'].notna()
]

# --- CREATE SYNTHETIC STATION ID ---
raw_df['station_id'] = raw_df['EV Vendor'] + " - " + raw_df['address']

# --- INIT LLM-AWARE DATAFRAME ---
# Drop unhashable (list/dict) columns before SmartDataframe
hashable_df = raw_df.copy()
for col in hashable_df.columns:
    if hashable_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
        hashable_df = hashable_df.drop(columns=[col])

df = SmartDataframe(hashable_df, config={"llm": llm})

# --- APP LAYOUT ---
#st.set_page_config(page_title="EV Review Insights", layout="wide")
st.title("🔌 EV Charging Station Review Explorer (MVP v1)")

# --- SIDEBAR INPUT ---
st.sidebar.header("Ask a Question or Choose Analysis")

# --- Define sample prompts ---
sample_prompts = [
    "Show a list of vendors along with the total number of charging stations they have installed.",
    "Which electric vehicle charging station vendor has the highest number of stations in the dataset?",
    "Count the number of EV charging stations in the zip code 95110."
]

# --- Select example to populate ---
with st.sidebar.expander("💡 Select an Example Prompt", expanded=False):
    selected_example = st.selectbox("Choose a sample prompt", [""] + sample_prompts, key="example_select")
    if st.button("📋 Use this Example"):
        st.session_state.user_query = selected_example

# --- Form for user input ---
with st.sidebar.form("user_query_form"):
    user_query = st.text_area(
        "User Query",
        value=st.session_state.get("user_query", ""),
        placeholder="Type your query here...",
        height=100
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.user_query = user_query

# Refine User Prompt 
client = OpenAI(api_key=st.secrets["OpenAI_API_KEY"])
def refine_prompt(user_prompt):
    system_msg = (
        "You are an EV analytics expert. "
        "Refine the user query to be precise and relevant for a dataframe analysis on electric vehicle charging station reviews. "
        "If the prompt implies a chart or graph, ask for X and Y values. Avoid vague terms. Respond with only the refined query."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# --- USER QUERY TO GPT-4o ---
if user_query:
    st.subheader("🤖 LLM Answer")
    try:
        #answer = df.chat(user_query)
        refined_prompt = refine_prompt(user_query)
        st.markdown(f"🔍 **Refined Prompt**: `{refined_prompt}`")
        answer = df.chat(refined_prompt)
        
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                border-left: 5px solid #0a84ff;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 8px;
                font-family: monospace;
                white-space: pre-wrap;
                color: #111111;
            ">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error: {e}")
        


# --- PREDEFINED FUNCTIONS ---
def get_top_stations():
    return raw_df.groupby('station_id')['reviewsCount'].sum().sort_values(ascending=False).head(10)

def get_worst_stations():
    return raw_df.groupby('station_id')['totalScore'].mean().sort_values().head(10)

def average_rating_by_vendor():
    return raw_df.groupby('EV Vendor')['totalScore'].mean().sort_values(ascending=False)

def review_distribution():
    col_name = 'reviewsDistribution.1'
    if col_name in raw_df.columns:
        return raw_df[col_name].value_counts().sort_index()
    else:
        return pd.Series(dtype=int)

def wait_time_mentions():
    return raw_df['reviews'].apply(lambda x: 'wait' in str(x).lower()).value_counts()

def stations_with_long_wait():
    return raw_df[raw_df['reviews'].apply(lambda x: 'long wait' in str(x).lower())]

def top_complaints():
    return df.chat("What are the top complaints across all reviews?")

def summarize_by_city():
    return df.chat("Summarize the best and worst EV stations by city.")

def vendor_sentiment():
    return df.chat("Compare sentiment for each vendor across all reviews.")

def peak_occupancy_analysis():
    return df.chat("What are the busiest times of day across EV stations?")

# --- RENDER PREDEFINED INSIGHTS ---
st.subheader("📊 Key Insights")

st.markdown("**Top 10 Stations by Review Volume**")
show_chart = st.toggle("Show as Bar Chart", value=False, key="top10_toggle_switch")
top10 = get_top_stations()
if not show_chart:
    st.dataframe(top10, use_container_width=True)
else:
    import altair as alt
    # If top10 is a Series, convert to DataFrame for charting
    if isinstance(top10, pd.Series):
        top10_df = top10.reset_index()
        top10_df.columns = ["Station", "Reviews"]
    else:
        top10_df = top10
    chart = alt.Chart(top10_df).mark_bar().encode(
        x=alt.X('Station:N', sort='-y', title='Station'),
        y=alt.Y('Reviews:Q', title='Review Count'),
        tooltip=['Station', 'Reviews']
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

st.markdown("**Worst 10 Stations by Avg Rating**")
st.dataframe(get_worst_stations(), use_container_width=True)

st.markdown("**Avg Rating by Vendor**")
st.dataframe(average_rating_by_vendor(), use_container_width=True)

st.markdown("**Review Distribution (1-star only)**")
dist_series = review_distribution()
if not dist_series.empty:
    st.bar_chart(dist_series)
else:
    st.info("No review distribution data available.")

st.markdown("**Mentions of Wait Time**")
st.dataframe(wait_time_mentions(), use_container_width=True)

st.markdown("**Stations with 'Long Wait' in Reviews**")
st.dataframe(stations_with_long_wait()[['station_id', 'address']].drop_duplicates(), use_container_width=True)


# --- BONUS INSIGHTS ---
st.subheader("✨ Bonus LLM Insights")
st.write("Top complaints across stations:")
st.write(top_complaints())

st.write("City-based Summary:")
st.write(summarize_by_city())

st.write("Vendor Sentiment Summary:")
st.write(vendor_sentiment())

st.write("Occupancy Patterns:")
st.write(peak_occupancy_analysis())
