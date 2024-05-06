# Disease Identification and Drug Recommendation using Large Language Models

Team Members
  - Kalaiarasi Kaliappan
  - Sai Charan Thummalapudi
  - Shobana Siranjeevilu

Project idea:

In today's rapidly evolving healthcare landscape, enhancing patient understanding and involvement in their own care is crucial for improving treatment outcomes. Our project aims to transform this aspect of healthcare by leveraging healthcare data and advanced language models, effectively utilizing the power of natural language processing, and fine-tuning large language models (LLMs). By utilizing traditional methodologies with cutting-edge technology, we intend to provide individuals with a reliable tool to better understand their health issues and the types of medications that may be suggested based on their symptoms. This initiative seeks to demystify medical information for the average person, thereby empowering them to make informed decisions about their health, ultimately contributing to a more informed public and efficient healthcare delivery.



Steps to run the application

## create a virtual env

 python3 -m venv healthcare_gpt   

## Activate virtual env

source healthcare_gpt/bin/activate

## Install the packages

pip3 install -r requirements.txt

## run the webapp

cd streamlit
python3 -m streamlit run web_app.py
