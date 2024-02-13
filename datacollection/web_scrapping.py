
import re
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")



def search_wikipedia_for_url(search_term):

    endpoint = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": search_term
    }
    
    response = requests.get(endpoint, params=params)
    
    if response.status_code == 200:
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            return search_results
        
    


def web_scrape(disease_list):

    """
    disease_list: input list takes list of disease to search and collect the symptoms from wikipedia
    diseases_with_symptoms: Returns dictionary containing diseases and its symptoms
    """
    diseases_with_symptoms = {}
    for disease in disease_list:
        url_link = search_wikipedia_for_url(disease)
        
        if url_link:
            for url in url_link:

                page_title = url['title']

                page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                
                wiki = requests.get(page_url, verify=False)
                soup = BeautifulSoup(wiki.content, 'html5lib')
                info_table = soup.find("table", {"class": "infobox"})
                if info_table:
                    for row in info_table.find_all("tr"):
                        header = row.find("th", {"scope": "row"})
                        if header and header.get_text() == "Symptoms":
                            symptom = row.find("td")
                            if symptom:
                                symptom = ' '.join(re.sub(r'<.*?>|\[.*?\]', ' ', str(symptom)).split())
                                diseases_with_symptoms[disease] = symptom

                if disease in diseases_with_symptoms.keys():
                    break

    return diseases_with_symptoms