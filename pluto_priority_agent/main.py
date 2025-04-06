import streamlit as st
st.set_page_config(page_title="Idea Ranking Dashboard", layout="wide")
import pandas as pd
import json
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from templates import temp, temp2
import plotly.express as px
import time


# --- Prompt templates ---
prompt_template = PromptTemplate.from_template(temp)
prompt_template2 = PromptTemplate.from_template(temp2)

# --- Cache LLM initialization ---
@st.cache_resource
def get_llm_and_parser():
    llm = OllamaLLM(model="llama3.2")
    parser = StrOutputParser()
    return llm, parser

# --- Cache ranking data generation ---
@st.cache_data(show_spinner="🔍 Scoring ideas with LLM...")
def generate_ranking_data(df):
    llm, parser = get_llm_and_parser()
    chain = (
        {
            "id": RunnablePassthrough(),
            "title": RunnablePassthrough(),
            "description": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | parser
    )

    result_keys = [
        "value_created_score", "user_demand_score", "business_impact_score",
        "effort_time_score", "effort_resources_score", "effort_dependencies_score",
        "feasibility_concerns_score", "risk_score", "uncertainties_score",
        "similar_tech_score", "user_adoption_score", "thought"
    ]
    result_df = pd.DataFrame(columns=result_keys)

    for _, row in df.iterrows():
        input_data = {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
        }
        response = chain.invoke(input_data)
        parsed = json.loads(response)

        s1 = parsed["s1"]
        impact = s1["impact_metrics"]
        effort = s1["effort_metrics"]
        other = s1["other_metrics"]
        final = parsed["final_output"]

        result = {
            "value_created_score": impact["value_created"]["score"],
            "user_demand_score": impact["user_demand"]["score"],
            "business_impact_score": impact["business_impact"]["score"],
            "effort_time_score": effort["time"]["score"],
            "effort_resources_score": effort["resources"]["score"],
            "effort_dependencies_score": effort["dependencies"]["score"],
            "feasibility_concerns_score": other["feasibility_concerns"]["score"],
            "risk_score": other["risk"]["score"],
            "uncertainties_score": other["uncertainties_or_gaps"]["score"],
            "similar_tech_score": other["similar_tech_performance"]["score"],
            "user_adoption_score": other["user_adoption"]["score"],
            "thought": final["thought"]
        }

        result_df = pd.concat([result_df, pd.DataFrame([result])], ignore_index=True)

    result_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    return result_df

# --- Cache reasoning generation ---
@st.cache_data(show_spinner="💬 Generating summary from LLM...")
def generate_reasoning(df, impact_weight, effort_weight, other_weight):
    llm, parser = get_llm_and_parser()
    chain = (
        {
            "impact_weight": RunnablePassthrough(),
            "effort_weight": RunnablePassthrough(),
            "other_weight": RunnablePassthrough(),
            "json_data": RunnablePassthrough()
        }
        | prompt_template2
        | llm
        | parser
    )

    json_data = df.to_dict(orient='records')

    input_data = {
        "impact_weight": impact_weight,
        "effort_weight": effort_weight,
        "other_weight": other_weight,
        "json_data": json_data
    }

    response = chain.invoke(input_data)
    return response

# --- Pure function for rank score ---
def calculate_rank_score(row, impact_weight, effort_weight, other_weight):
    impact_score = (
        row['value_created_score'] +
        row['user_demand_score'] +
        row['business_impact_score']
    ) / 3

    effort_score = (
        (10 - row['effort_time_score']) +
        (10 - row['effort_resources_score']) +
        (10 - row['effort_dependencies_score'])
    ) / 3

    other_score = (
        (10 - row['feasibility_concerns_score']) +
        (10 - row['risk_score']) +
        (10 - row['uncertainties_score']) +
        row['similar_tech_score']
    ) / 4

    impact_score_normalized = impact_score / 10
    effort_score_normalized = effort_score / 10
    other_score_normalized = other_score / 10

    rank_score = (
        impact_score_normalized * impact_weight +
        effort_score_normalized * effort_weight +
        other_score_normalized * other_weight
    )

    return round(rank_score, 4), impact_score, effort_score, other_score

# --- Streamlit App ---
def main():
    st.title("🚀 Idea Prioritization Dashboard")
    uploaded_file = st.file_uploader("📂 Upload your scored CSV file", type=["csv"])

    st.markdown("### Adjust Weights to Rank Ideas (must sum to 1.0)")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        impact_weight = st.slider("🔷 Impact", 0.0, 1.0, 0.4, step=0.05)

    with col2:
        effort_weight = st.slider("🔶 Ease of Implementation", 0.0, 1.0, 0.3, step=0.05)

    with col3:
        other_weight = st.slider("🟡 Reliability / Confidence", 0.0, 1.0, 0.3, step=0.05)

    total = round(impact_weight + effort_weight + other_weight, 2)
    if total != 1.0:
        st.error(f"Total weight must equal 1.0 — currently {total}")
        return


    if uploaded_file:
        if st.button("▶️ Run Prioritization"):
            df = pd.read_csv(uploaded_file)
            
            df_scored = generate_ranking_data(df)

            df_scored[['rank_score', 'Impact', 'Ease of Implementation', 'Reliability/Confidence/Resilience']] = df_scored.apply(
                lambda x: pd.Series(calculate_rank_score(x, impact_weight, effort_weight, other_weight)),
                axis=1
            )

            df_top3 = df_scored.sort_values(by="rank_score", ascending=False).head(3)

            st.subheader("🏆 Top 3 Ideas with Thoughts")
            with st.spinner("⏳ Processing... Please wait."):
                time.sleep(5)
            for _, row in df_top3.iterrows():
                with st.expander(f"💡 {row['title']} (Rank Score: {round(row['rank_score'], 3)})"):
                    st.markdown(
                        f"""
                        <div style='font-size: 0.92rem; line-height: 1.6;'>
                            <p><strong>🧾 Description:</strong><br>{row['description']}</p>
                            <p><strong>🧠 LLM Thought:</strong><br>{row['thought'].strip()}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            summary = generate_reasoning(df_top3, impact_weight, effort_weight, other_weight)
            st.subheader("💬 Reasoning")
            st.markdown(
                f"<div style='white-space: pre-wrap; font-size: 0.95rem; background-color: var(--background-secondary); padding: 10px; border-radius: 8px;'>{summary.strip()}</div>",
                unsafe_allow_html=True
            )

            st.subheader("📊 3D Scatter Plot of Ideas")
            df_viz = df_scored.copy()
            top_3_ids = df_top3["id"].tolist()
            df_viz["Category"] = df_viz["id"].isin(top_3_ids).map({True: "Top 3 Idea", False: "Other Idea"})

            fig = px.scatter_3d(
                df_viz,
                x="Impact",
                y="Reliability/Confidence/Resilience",
                z="Ease of Implementation",
                color="Category",
                color_discrete_map={
                    "Top 3 Idea": "#1f77b4",      # Dark Blue
                    "Other Idea": "#aec7e8"       # Light Blue
                },
                hover_data=["title", "Impact", "Ease of Implementation", "Reliability/Confidence/Resilience"],
                title="3D Scatter Plot of Healthcare Tech Ideas",
                labels={
                    "Impact": "Impact",
                    "Ease of Implementation": "Ease of Implementation",
                    "Reliability/Confidence/Resilience": "Reliability/Confidence/Resilience"
                }
            )


            fig.update_layout(
                legend_title_text='Idea Visualization',
                width=1000,
                height=800,
                scene=dict(
                    xaxis=dict(range=[0, 10], title="Impact"),
                    yaxis=dict(range=[0, 10], title="Reliability/Confidence/Resilience"),
                    zaxis=dict(range=[0, 10], title="Ease of Implementation")
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📄 Please upload a scored CSV file to continue.")

if __name__ == "__main__":
    main()
