import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
# Use a pipeline as a high-level helper
#from transformers import pipeline

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text):
    # Load model directly
    ### LLama2 model
    llm=CTransformers(model='/Users/saicharanthummalapudi/Desktop/healthgpt_st_app/models/pytorch_model-00002-of-00002.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    ## Prompt Template

    template="""
        <s>[INST] {symptom} [/INST] {disease} </s>
            """
    
    prompt=PromptTemplate(input_variables=["input_text"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(input_text=input_text))
    print(response)
    return response






st.set_page_config(page_title="HealthGPT",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Disease Prediction and Drug Recommendation")

input_text=st.text_input("Enter the Symptom")
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text))