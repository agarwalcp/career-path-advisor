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

## 👥 Team & Project Division

This project is divided between two key roles to ensure efficient development and collaboration.

### Member 1: Machine Learning & Data Scientist (Branch: `ml-data-science`)
**Primary Focus:** Data processing, model training, and NLP logic.
- **Files:** `career_path_project_final_nlp.ipynb`, `career_data.csv`
- **Responsibilities:** Clean the data, train the TF-IDF and BERT models, and improve the skill extraction algorithms.

### Member 2: Web App Developer & UI/UX (Branch: `web-app-ui`)
**Primary Focus:** Streamlit application, user interface, and model integration.
- **Files:** `career_advisor_app.py`, `requirements.txt`
- **Responsibilities:** Build out the Streamlit interface, integrate the ML models, manage state, and style the application.

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

### 3. Running the Machine Learning Notebook (Member 1)
Open the Jupyter Notebook to train or evaluate the models:
```bash
jupyter notebook career_path_project_final_nlp.ipynb
```

### 4. Running the Web App (Member 2)
To start the Streamlit application locally:
```bash
streamlit run career_advisor_app.py
```

## 🤝 How to Collaborate
1. Make sure you are on your designated branch (`ml-data-science` or `web-app-ui`).
2. Commit your changes: `git commit -m "Your descriptive message"`
3. Push to your branch: `git push origin <your-branch-name>`
4. Open a Pull Request on GitHub to merge your features into the `main` branch.
