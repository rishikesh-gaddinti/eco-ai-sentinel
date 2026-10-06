import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class EnvironmentalAgent:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-3.5-flash')
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = self._setup_knowledge_base()
        
    def _setup_knowledge_base(self):
        docs = [
            Document(page_content="High levels of PM10 and PM2.5 in urban areas are typically caused by heavy diesel traffic, nearby construction sites, or blowing dust.", metadata={"source": "Urban Air Quality Journal"}),
            Document(page_content="A sudden spike in Carbon Monoxide (CO) combined with Nitrogen Dioxide (NO2) often indicates a localized traffic bottleneck or an industrial combustion event.", metadata={"source": "EPA Guidelines"}),
            Document(page_content="Elevated Ground-level Ozone is usually formed by the reaction of VOCs and NOx in the presence of intense sunlight, often peaking in the late afternoon.", metadata={"source": "Meteorological Research"}),
            Document(page_content="An isolated spike in PM2.5 without an increase in other gases strongly suggests a localized fire or smoke drifting from a distant wildfire.", metadata={"source": "Wildfire Monitoring Institute"})
        ]
        return Chroma.from_documents(docs, self.embeddings)

    def analyze_anomaly(self, sensor_values, anomaly_score, provider="Gemini", groq_key=None):
        query = f"Causes of air pollution involving these values: {sensor_values}"
        docs = self.vector_store.similarity_search(query, k=2)
        research_context = "\n".join([d.page_content for d in docs])
        
        prompt = f"""
        You are an expert Environmental Data Scientist. An anomaly was detected in the sensor network.
        
        Current Sensor Readings: {sensor_values}
        LSTM Autoencoder Anomaly Score: {anomaly_score}
        
        Relevant Environmental Research Database (RAG Context):
        {research_context}
        
        Based on the sensor readings and the provided research context, write a short, scientific report (max 2 paragraphs) explaining the potential real-world cause of this anomaly. Be concise and authoritative.
        """
        
        if provider == "Groq":
            if not groq_key:
                return "Error: Groq API Key is missing. Please configure it in the sidebar settings."
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="qwen/qwen3.8-27b",
                    temperature=0.3
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                return f"Groq Error: {str(e)}"
        else:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Gemini Error: {str(e)}"
