# 🔭 CareerLens AI - Career Path Advisor

An AI-powered career recommendation system that analyzes resumes using Natural Language Processing (NLP) to predict suitable career paths, identify skill gaps, and provide tailored learning roadmaps.

## 🌟 Features
- **Smart Career Matching:** Uses NLP and Sentence Transformers (BERT) to analyze resume semantics and match them with over 20+ career profiles.
- **Skill Gap Analysis:** Highlights core and advanced skills you have, and those you are missing.
- **Learning Roadmaps:** Provides curated courses, certifications, and resources to help you bridge your skill gaps.
- **Job Description Matcher:** Compares your resume against a specific job description to calculate a match percentage.

## 🛠️ Technology Stack
- **Frontend / Web App:** Streamlit, Pandas, custom CSS
- **Machine Learning / NLP:** Scikit-learn (Logistic Regression, TF-IDF), Sentence-Transformers (BERT), NLTK
- **Data:** Resume datasets from Kaggle (`career_data.csv`)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/agarwalcp/career-path-advisor.git
cd career-path-advisor
```

### 2. Install dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Note: You may also need to install `pdfplumber` if you wish to process PDF resumes.)*

### 3. Exploring the Machine Learning Model
The ML models and text preprocessing code can be found and experimented with inside the Jupyter Notebook:
```bash
jupyter notebook career_path_project_final_nlp.ipynb
```

### 4. Running the Web App
To start the Streamlit application locally and interact with the UI:
```bash
streamlit run career_advisor_app.py
```
Open the provided `localhost` URL in your browser to view the application.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or create a pull request if you have improvements for the recommendation system or the user interface.
