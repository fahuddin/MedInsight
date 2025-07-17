# 🧠 MedInsight: AI-Powered Clinical Risk Prediction Platform

**MedInsight** is a full-stack AI system that predicts hospital readmissions and chronic condition risks using structured EHR data and unstructured clinical notes. It integrates Machine Learning, NLP, and Large Language Models (LLMs) to generate risk predictions, explanations, and patient summaries for clinical decision support.

---

## 🚀 Features

- 📊 **ETL Pipelines** for structured and unstructured clinical data  
- 🧠 **ML Models** for risk prediction (e.g., readmission)  
- 🗣️ **NLP Pipelines** for cleaning and embedding clinical notes  
- 🤖 **LLM Integration** for patient summarization (RAG w/ GPT or Cohere)  
- 📡 **REST API** (FastAPI) to serve predictions and explanations  
- 📈 **Dashboard** (Streamlit) for interactive visualization  
- 🧪 **Testing Suite** for pipelines, models, and endpoints  

---

## 🧱 Project Structure

MedInsight/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_nlp_pipeline.ipynb
│   └── 04_llm_experiments.ipynb
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── nlp/
│   ├── llm/
│   ├── api/
│   └── dashboard/
├── models/
├── scripts/
├── tests/
├── reports/
├── requirements.txt
├── docker-compose.yml
└── README.md


# Clone the repository
git clone https://github.com/fahuddin/MedInsight.git
cd MediInsight

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
