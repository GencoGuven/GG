from flask import Flask, render_template, request
import requests
from huggingface_hub import InferenceClient

#biopython

app = Flask(__name__)

hf_token = "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"
google_api_key = "AIzaSyD8khN_rd4QkZaSs8YJUs9Umw177TYyvxA"
google_cse_id = "14ab3685d90c34692"

def search_mayoclinic(query, site="mayoclinic.org/diseases-conditions/"):
    print(f"Searching Mayo Clinic for: {query}")
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": f"{query} site:{site}"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "items" in data:
            return [item["snippet"] for item in data["items"]]
        else:
            print("No results found.")
            return []
    else:
        print(f"Error: {response.status_code}")
        return []

def search_google(query):
    print(f"Searching Google for: {query}")
    base_url = "https://www.googleapis.com/customsearch/v1"
    medical_query = f"{query} site:medlineplus.gov OR site:mayoclinic.org OR site:nih.gov OR site:webmd.com OR site:healthline.com OR site:genecards.org OR site:www.ncbi.nlm.nih.gov OR site:nhs.uk"
    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": medical_query
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "items" in data:
            return [item["snippet"] for item in data["items"]]
        else:
            print("No search results found.")
            return []
    else:
        print(f"Error: {response.status_code}")
        return []

def search_disease(symptoms):
    mayoclinic_results = search_mayoclinic(symptoms)
    google_results = search_google(symptoms)

    combined_results = google_results

    mayoclinic_length = sum(len(item) for item in mayoclinic_results)
    if mayoclinic_length < 500:
        return "No diseases are found for this symptom."
    return generate_chatbot_response(" ".join(combined_results), symptoms)

def generate_chatbot_response(combined_abstracts, symptom):
    try:
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
        prompt = (
            f"Using the following abstracts, provide a detailed explanation about the top 5 most frequent diseases "
            f"that this symptom '{symptom}' can cause without stating that they are the top 5 in frequency. Don't talk about what caused this symptom '{symptom}'. Explain how '{symptom}' can be a symptom of these diseases "
            f"its possible impact on the patient's quality of life. Briefly provide guidance or advice on what they might consider or do next. "
            f"Do not mention the abstracts or articles themselves. Focus on giving information about the top 5 most frequent diseases in a conversational manner without stating that you will discuss these diseases.\n\n"
            f"{combined_abstracts}\n\n"
            "Summarize the key findings and provide a patient-centered response that connects the symptom to the diseases described. "
            "Avoid mentioning the articles or abstracts. Focus on the diseases and provide useful information in a friendly and approachable tone."
        )

        response = client.text_generation(prompt=prompt, max_new_tokens=1000)
        return response['generated_text'] if 'generated_text' in response else response
    except Exception as e:
        return f"Error: {e}"

def search_disease_genes(disease):
    mayoclinic_results = search_mayoclinic(disease)
    google_results = search_google(disease)

    combined_results = google_results

    mayoclinic_length = sum(len(item) for item in mayoclinic_results)
    if mayoclinic_length < 500:
        return "No diseases are found for this symptom."
    return generate_chatbot_response_genes(" ".join(combined_results), disease)

def generate_chatbot_response_genes(combined_abstracts, disease):
    try:
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
        prompt = (
            f"Using the following abstracts, provide a detailed list of all the genes and bacteria related to the disease '{disease}'. "
            f"Include the following in your response:\n"
            f"- A list of top 5 most relevant genes associated with the disease, with a brief description of their role.\n"
            f"- A list of top 5 most relevant bacteria related to the disease, with a brief explanation of their impact.\n"
            f"Briefly explain how these genes and bacteria could be connected to this disease and briefly discuss their potential impact on the patient's health. "
            f"Focus on providing a comprehensive and clear list of information.\n\n"
            f"{combined_abstracts}\n\n"
            "Summarize the key findings and provide a detailed list of genes and bacteria associated with the disease. "
            "Do not mention the abstracts or articles themselves. Concentrate on delivering a complete list of relevant genes and bacteria."
        )
        response = client.text_generation(prompt=prompt, max_new_tokens=1000)
        return response['generated_text'] if 'generated_text' in response else response
    except Exception as e:
        return f"Error: {e}"

@app.route('/', methods=['GET', 'POST'])
def home():
    chatbot_response = None
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        disease = request.form.get('disease')
        if symptoms:
            chatbot_response = search_disease(symptoms)
        elif disease:
            chatbot_response = search_disease_genes(disease)
    return render_template('index3.html', response=chatbot_response)

if __name__ == '__main__':
    app.run(debug=True)
