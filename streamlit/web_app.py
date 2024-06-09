import streamlit as st
import os
from transformers import pipeline
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

## Function To get response from Language model

def get_disease(symptoms):

    classifier = pipeline("text-classification", model="shobana/myproject")
    output = classifier(symptoms)

    return output[0]['label']

def get_drug_recommendation(disease):
    
    OpenAI_api_key = ""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=OpenAI_api_key)
    ddg_search = DuckDuckGoSearchRun()
    tools = [
            Tool(
                name="DuckDuckGo Search",
                func=ddg_search.run,
                description="Useful to browse information from the Internet.",
            )
            ]
    agent = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True
            )
    
    result = agent.run(
        "Find recent medication used for {}".format(disease)
        )
    

    return result


st.set_page_config(page_title="HealthGPT",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Disease Prediction and Drug Recommendation")

input_text=st.text_input("Enter the Symptom")
    
submit=st.button("Generate")

## Final response
if submit:
    disease = get_disease(input_text)

    # Note: Commented drug recommendation function because LLM agent used for \
                # incuding browsing capabilities has exceeded the rate limits.
    
    #drug = get_drug_recommendation(disease)

    response = {
        'disease': disease,
        'drugs recommended': []
    }


    st.write(response)