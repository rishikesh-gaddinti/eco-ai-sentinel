## Goal Description
Build an **LLM-Powered Environmental Anomaly Agent** for a capstone project and IEEE conference paper. 
The system will ingest real-time environmental data (e.g., Air Quality/Weather) from live APIs. It will use an advanced Deep Learning model (LSTM Autoencoder) to detect statistical anomalies, and trigger an LLM (Gemini/Groq) equipped with Retrieval-Augmented Generation (RAG) to automatically analyze and explain the potential causes of the anomaly. 
The goal includes building a deployable real-time prototype (featuring both a Live Mode and a Demo Mode) and drafting a high-quality IEEE research paper based on this architecture.

## Deployment Feasibility
**Chances of deployment: 100%.** 
We will use free-tier platforms suitable for academic prototypes. I will guide you step-by-step through the entire deployment process when we get there.
- **Frontend/Dashboard:** Streamlit Community Cloud (Free, easy to deploy from GitHub).
- **Backend/Data Pipeline:** Python scripts running directly within the Streamlit app.
- **Database/Vector Store:** ChromaDB (runs locally in-memory, no server needed).
- **LLM:** Gemini API (via Google AI Studio free tier) or Groq API. 

## User Review Required
> [!IMPORTANT]
> The plan has been updated to reflect your great ideas! Please give it a final review. 
> If everything looks good, just give me the signal, and we will begin Phase 1 development!

## Proposed Architecture and Roadmap

### Phase 1: Core System Development (Local Prototype)
- **Data Ingestion (Real-Time API):** Write scripts to fetch real-time Air Quality/Weather data from free open APIs (e.g., OpenWeatherMap, Open-Meteo, or OpenAQ).
- **Advanced Anomaly Detection Module:** Instead of a basic model, we will build an **LSTM Autoencoder** using TensorFlow/Keras or PyTorch. This is an advanced Deep Learning architecture for time-series anomaly detection and will look much stronger and more rigorous in your IEEE paper.
- **RAG & LLM Engine:** Set up `LangChain`. Create a local vector database (`ChromaDB`) loaded with environmental research context. When the LSTM flags an anomaly, the LLM will generate a contextual report.

### Phase 2: Web Dashboard Development (Dual Mode)
- **Dashboard:** Build a real-time web application using `Streamlit`.
- **The "Dual Mode" Toggle (Crucial for Demo):** We will add a UI switch in the app with two sections:
    - **Live Real-Time Mode:** The app purely monitors the real-time API data. This proves to your faculty that the cloud architecture is connected to live real-world data streams.
    - **Simulation/Demo Mode:** The app takes the real-time data but mathematically injects a "synthetic spike" (e.g., suddenly multiplying pollution metrics by 5). This guarantees you can trigger an anomaly and show off the LLM's response live during your short 10-minute presentation.

### Phase 3: Cloud Deployment (Fully Guided)
- **Packaging:** Create a `requirements.txt` file for our dependencies.
- **Hosting:** We will deploy the web dashboard to **Streamlit Community Cloud**. I will provide you with the exact button clicks and steps required to connect your GitHub repository to Streamlit. It is very simple and requires no prior cloud experience!
- **Result:** You will have a live, public URL (e.g., `your-capstone.streamlit.app`) to put in your presentation slides.

### Phase 4: IEEE Paper Writing
- **Abstract & Intro:** Frame the problem of interpreting high-volume environmental alerts.
- **Literature Review:** Discuss the transition from basic thresholds to advanced Deep Learning (LSTM Autoencoders) combined with GenAI.
- **Proposed Architecture:** Generate high-quality diagrams of our dual-mode cloud setup.
- **Results & Evaluation:** Benchmark the LSTM accuracy and the GenAI explanation quality.

## Verification Plan

### Automated Tests
- Python scripts to ensure data is successfully fetched from the real-time public APIs.

### Manual Verification
- We will toggle on "Simulation Mode" to manually trigger a data spike.
- Verify that the dashboard graph updates immediately and the LSTM flags the anomaly.
- Verify that the Gemini/Groq LLM generates a coherent, scientifically plausible explanation within 5-10 seconds.
- Toggle back to "Live Real-Time Mode" to ensure the real data streams correctly.
