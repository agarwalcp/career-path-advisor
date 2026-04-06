"""
Career Path Advisor — Streamlit App (Fixed & Enhanced)
Based on NLP project: TF-IDF + BERT + Cosine Similarity career recommendation
Features:
  - Real-time resume upload (PDF/TXT/paste)
  - Career path prediction (TF-IDF ML)
  - BERT semantic similarity recommendations
  - Skill gap analysis with animated progress bars
  - Personalized learning roadmap with curated resources
  - JD (Job Description) matcher
  - Animated skill radar chart
  - Dark sidebar with gradient hero
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Career Path Advisor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,111,247,.25) 0%, transparent 70%);
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 35%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(167,139,250,.15) 0%, transparent 70%);
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.9rem;
    margin: 0 0 .5rem;
    line-height: 1.1;
    position: relative; z-index: 1;
  }
  .hero p {
    font-size: 1.05rem;
    opacity: .78;
    margin: 0;
    font-weight: 300;
    position: relative; z-index: 1;
  }
  .hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: .75rem; font-weight: 600;
    letter-spacing: .06em;
    color: rgba(255,255,255,.9);
    margin-bottom: 1rem;
    position: relative; z-index: 1;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: .68rem; font-weight: 700;
    letter-spacing: .14em; text-transform: uppercase;
    color: #7c6ff7; margin-bottom: .35rem;
  }
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.55rem; margin: 0 0 1.2rem;
    color: #1a1a2e;
  }

  /* ── Career cards ── */
  .career-card {
    background: white;
    border: 1.5px solid #ede9ff;
    border-radius: 16px;
    padding: 1.15rem 1.4rem;
    margin-bottom: .8rem;
    position: relative;
    transition: box-shadow .25s, transform .2s;
  }
  .career-card:hover {
    box-shadow: 0 8px 28px rgba(124,111,247,.18);
    transform: translateY(-2px);
  }
  .career-card.top {
    border-color: #7c6ff7;
    background: linear-gradient(135deg, #fdfcff 0%, #f3f0ff 100%);
    box-shadow: 0 4px 18px rgba(124,111,247,.12);
  }
  .career-rank {
    position: absolute; top: 1rem; right: 1.1rem;
    font-size: .7rem; font-weight: 700; color: #7c6ff7;
    background: #ede9ff; padding: 3px 10px; border-radius: 99px;
  }
  .career-rank.gold { background: #fef9c3; color: #92400e; }
  .career-name { font-weight: 700; font-size: 1.05rem; color: #1a1a2e; margin-bottom: .2rem; }
  .sim-bar-bg { height: 7px; background: #ede9ff; border-radius: 4px; margin-top: .5rem; overflow: hidden; }
  .sim-bar {
    height: 7px; border-radius: 4px;
    background: linear-gradient(90deg, #7c6ff7, #a78bfa, #c4b5fd);
    transition: width .8s cubic-bezier(.4,0,.2,1);
  }

  /* ── Skill pills ── */
  .skill-have {
    display: inline-block; background: #dcfce7; color: #166534;
    border-radius: 99px; padding: 3px 13px; font-size: .8rem; font-weight: 500; margin: 3px;
  }
  .skill-missing {
    display: inline-block; background: #fee2e2; color: #991b1b;
    border-radius: 99px; padding: 3px 13px; font-size: .8rem; font-weight: 500; margin: 3px;
  }
  .skill-learn {
    display: inline-block; background: #fef3c7; color: #92400e;
    border-radius: 99px; padding: 3px 13px; font-size: .8rem; font-weight: 500; margin: 3px;
  }
  .skill-neutral {
    display: inline-block; background: #f1f5f9; color: #475569;
    border-radius: 99px; padding: 3px 13px; font-size: .8rem; font-weight: 500; margin: 3px;
  }

  /* ── Metric tiles ── */
  .metric-tile {
    background: white;
    border: 1.5px solid #ede9ff;
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: box-shadow .2s;
  }
  .metric-tile:hover { box-shadow: 0 4px 16px rgba(124,111,247,.12); }
  .metric-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; color: #7c6ff7; line-height: 1;
  }
  .metric-label { font-size: .76rem; color: #64748b; font-weight: 500; margin-top: .3rem; }

  /* ── Step badge ── */
  .step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #7c6ff7, #a78bfa);
    color: white; font-size: .8rem; font-weight: 700;
    margin-right: .6rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(124,111,247,.35);
  }

  /* ── Resource cards ── */
  .resource-card {
    background: white;
    border: 1px solid #e9ecef;
    border-left: 4px solid #7c6ff7;
    border-radius: 0 12px 12px 0;
    padding: .9rem 1.1rem;
    margin-bottom: .6rem;
    transition: border-left-color .2s, box-shadow .2s;
  }
  .resource-card:hover {
    border-left-color: #a78bfa;
    box-shadow: 0 3px 12px rgba(124,111,247,.1);
  }
  .resource-title { font-weight: 600; font-size: .9rem; color: #1a1a2e; }
  .resource-meta  { font-size: .78rem; color: #64748b; margin-top: 3px; }

  /* ── JD match bar ── */
  .jd-bar-bg { height: 12px; background: #f1f5f9; border-radius: 6px; overflow: hidden; margin: .4rem 0; }
  .jd-bar {
    height: 12px; border-radius: 6px;
    background: linear-gradient(90deg, #7c6ff7, #a78bfa);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background: linear-gradient(160deg, #0f0c29, #1a1740) !important; }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,.88) !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }
  [data-testid="stSidebar"] .stSlider .stMarkdown { color: rgba(255,255,255,.6) !important; }
  [data-testid="stSidebar"] .stRadio label { color: rgba(255,255,255,.85) !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] { font-weight: 500; font-size: .92rem; padding-bottom: .5rem; }
  .stTabs [aria-selected="true"] { color: #7c6ff7 !important; border-bottom-color: #7c6ff7 !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c6ff7, #a78bfa) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: .65rem 2.2rem !important;
    transition: all .25s !important;
    box-shadow: 0 4px 14px rgba(124,111,247,.35) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,111,247,.45) !important;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploadDropzone"] {
    background: #faf9ff !important;
    border: 2px dashed #c4b5fd !important;
    border-radius: 16px !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader { font-weight: 500 !important; font-size: .9rem !important; }

  /* ── Info / warning ── */
  .stAlert { border-radius: 12px !important; }

  /* ── Hide footer ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #7c6ff7 !important; }

  /* ── Divider ── */
  hr { border: none; border-top: 1.5px solid #f1f0ff; margin: 1.5rem 0; }

  /* ── Progress bar override ── */
  .stProgress > div > div > div { background: linear-gradient(90deg, #7c6ff7, #a78bfa) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Skill Knowledge Base ───────────────────────────────────────────────────────
CAREER_SKILLS = {
    "Data Science": {
        "core":     ["python", "machine learning", "statistics", "pandas", "numpy", "scikit-learn", "sql", "data visualization", "matplotlib", "seaborn"],
        "advanced": ["deep learning", "tensorflow", "pytorch", "spark", "hadoop", "mlflow", "feature engineering", "a/b testing", "bayesian statistics"],
        "tools":    ["jupyter", "tableau", "power bi", "git", "docker", "aws", "gcp", "azure"],
        "resources": [
            {"title": "Python for Data Science Handbook", "type": "📘 Book", "link": "https://jakevdp.github.io/PythonDataScienceHandbook/"},
            {"title": "fast.ai – Practical Deep Learning", "type": "🎓 Course", "link": "https://course.fast.ai"},
            {"title": "Kaggle Learn", "type": "🏋️ Platform", "link": "https://www.kaggle.com/learn"},
            {"title": "StatQuest with Josh Starmer", "type": "▶️ YouTube", "link": "https://www.youtube.com/c/joshstarmer"},
        ]
    },
    "Machine Learning": {
        "core":     ["python", "machine learning", "scikit-learn", "tensorflow", "pytorch", "numpy", "pandas", "linear algebra", "calculus", "statistics"],
        "advanced": ["neural networks", "transformers", "nlp", "computer vision", "reinforcement learning", "model deployment", "mlops"],
        "tools":    ["jupyter", "git", "docker", "kubernetes", "mlflow", "weights & biases", "hugging face"],
        "resources": [
            {"title": "Hands-On ML with Scikit-Learn (Géron)", "type": "📘 Book", "link": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"},
            {"title": "CS229 Stanford ML Course", "type": "🎓 Course", "link": "https://cs229.stanford.edu"},
            {"title": "Hugging Face NLP Course", "type": "🎓 Course", "link": "https://huggingface.co/learn/nlp-course"},
            {"title": "Papers With Code", "type": "🔬 Research", "link": "https://paperswithcode.com"},
        ]
    },
    "Web Designing": {
        "core":     ["html", "css", "javascript", "responsive design", "figma", "ux design", "ui design", "typography", "color theory"],
        "advanced": ["react", "vue", "angular", "typescript", "tailwind", "framer motion", "animation", "accessibility", "web performance"],
        "tools":    ["figma", "adobe xd", "sketch", "webpack", "vite", "git", "vercel", "netlify"],
        "resources": [
            {"title": "The Odin Project", "type": "🎓 Course", "link": "https://www.theodinproject.com"},
            {"title": "CSS Tricks", "type": "📝 Blog", "link": "https://css-tricks.com"},
            {"title": "Refactoring UI", "type": "📘 Book", "link": "https://www.refactoringui.com"},
            {"title": "freeCodeCamp Responsive Design", "type": "🎓 Course", "link": "https://www.freecodecamp.org"},
        ]
    },
    "Java Developer": {
        "core":     ["java", "oop", "spring", "spring boot", "maven", "gradle", "sql", "rest api", "junit", "design patterns"],
        "advanced": ["microservices", "kafka", "redis", "docker", "kubernetes", "reactive programming", "jpa", "hibernate"],
        "tools":    ["intellij", "eclipse", "git", "jenkins", "sonarqube", "postman", "jira"],
        "resources": [
            {"title": "Effective Java – Joshua Bloch", "type": "📘 Book", "link": "https://www.oreilly.com/library/view/effective-java/9780134686097/"},
            {"title": "Spring Framework Docs", "type": "📄 Docs", "link": "https://spring.io/guides"},
            {"title": "Baeldung Java Tutorials", "type": "📝 Blog", "link": "https://www.baeldung.com"},
        ]
    },
    "Python Developer": {
        "core":     ["python", "oop", "django", "flask", "fastapi", "rest api", "sql", "postgresql", "testing", "git"],
        "advanced": ["async python", "celery", "redis", "docker", "kubernetes", "graphql", "sqlalchemy", "pydantic"],
        "tools":    ["vscode", "pycharm", "git", "docker", "aws", "gcp", "postman", "pytest"],
        "resources": [
            {"title": "Fluent Python", "type": "📘 Book", "link": "https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/"},
            {"title": "Real Python", "type": "🏋️ Platform", "link": "https://realpython.com"},
            {"title": "FastAPI Docs", "type": "📄 Docs", "link": "https://fastapi.tiangolo.com"},
        ]
    },
    "DevOps Engineer": {
        "core":     ["linux", "bash", "docker", "kubernetes", "ci/cd", "git", "terraform", "ansible", "jenkins", "monitoring"],
        "advanced": ["helm", "istio", "prometheus", "grafana", "elk stack", "vault", "service mesh", "gitops", "argocd"],
        "tools":    ["aws", "gcp", "azure", "github actions", "gitlab ci", "datadog", "pagerduty"],
        "resources": [
            {"title": "The DevOps Handbook", "type": "📘 Book", "link": "https://www.amazon.com/DevOps-Handbook/dp/1942788002"},
            {"title": "KodeKloud", "type": "🏋️ Platform", "link": "https://kodekloud.com"},
            {"title": "AWS Free Tier", "type": "☁️ Platform", "link": "https://aws.amazon.com/free"},
        ]
    },
    "Database": {
        "core":     ["sql", "postgresql", "mysql", "mongodb", "indexing", "normalization", "transactions", "stored procedures", "query optimization"],
        "advanced": ["redis", "cassandra", "elasticsearch", "data warehousing", "etl", "dbt", "snowflake", "bigquery", "replication"],
        "tools":    ["dbeaver", "pgadmin", "mongodb compass", "datagrip", "apache airflow"],
        "resources": [
            {"title": "Use The Index, Luke!", "type": "📘 Book", "link": "https://use-the-index-luke.com"},
            {"title": "PostgreSQL Tutorial", "type": "🎓 Course", "link": "https://www.postgresqltutorial.com"},
            {"title": "MongoDB University", "type": "🎓 Course", "link": "https://university.mongodb.com"},
        ]
    },
    "Network Security Engineer": {
        "core":     ["networking", "tcp/ip", "firewalls", "vpn", "penetration testing", "vulnerability assessment", "siem", "linux", "cryptography"],
        "advanced": ["threat intelligence", "incident response", "zero trust", "soc", "osint", "malware analysis", "forensics", "devsecops"],
        "tools":    ["wireshark", "nmap", "metasploit", "burp suite", "splunk", "nessus", "kali linux"],
        "resources": [
            {"title": "TryHackMe", "type": "🏋️ Platform", "link": "https://tryhackme.com"},
            {"title": "CompTIA Security+ Study Guide", "type": "📘 Book", "link": "https://www.comptia.org/certifications/security"},
            {"title": "Cybrary", "type": "🎓 Course", "link": "https://www.cybrary.it"},
        ]
    },
    "Hadoop": {
        "core":     ["hadoop", "mapreduce", "hdfs", "hive", "spark", "pig", "sqoop", "yarn", "java", "python"],
        "advanced": ["kafka", "hbase", "zookeeper", "storm", "flink", "delta lake", "iceberg", "data lakehouse"],
        "tools":    ["cloudera", "hortonworks", "aws emr", "databricks", "airflow", "nifi"],
        "resources": [
            {"title": "Hadoop: The Definitive Guide", "type": "📘 Book", "link": "https://www.oreilly.com/library/view/hadoop-the-definitive/9781491901687/"},
            {"title": "Databricks Academy", "type": "🏋️ Platform", "link": "https://academy.databricks.com"},
        ]
    },
    "ETL Developer": {
        "core":     ["sql", "python", "etl tools", "informatica", "talend", "data warehousing", "star schema", "data modeling", "ssis", "oracle"],
        "advanced": ["dbt", "airflow", "spark", "snowflake", "bigquery", "azure data factory", "real-time streaming", "kafka"],
        "tools":    ["informatica", "talend", "ssis", "airflow", "dbt", "snowflake", "azure data factory"],
        "resources": [
            {"title": "dbt Learn", "type": "🏋️ Platform", "link": "https://learn.getdbt.com"},
            {"title": "The Data Warehouse Toolkit", "type": "📘 Book", "link": "https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/"},
        ]
    },
    "Blockchain": {
        "core":     ["solidity", "ethereum", "web3.js", "smart contracts", "cryptography", "defi", "nft", "truffle", "hardhat", "javascript"],
        "advanced": ["layer 2 solutions", "zk proofs", "cross-chain bridges", "tokenomics", "dao governance", "ipfs", "rust (solana)"],
        "tools":    ["metamask", "remix", "hardhat", "truffle", "infura", "etherscan", "openzeppelin"],
        "resources": [
            {"title": "CryptoZombies", "type": "🏋️ Platform", "link": "https://cryptozombies.io"},
            {"title": "Ethereum Documentation", "type": "📄 Docs", "link": "https://ethereum.org/en/developers/docs/"},
            {"title": "Alchemy University", "type": "🎓 Course", "link": "https://university.alchemy.com"},
        ]
    },
    "Automation Testing": {
        "core":     ["selenium", "python", "java", "pytest", "testng", "rest assured", "api testing", "ci/cd", "git", "agile"],
        "advanced": ["cypress", "playwright", "appium", "performance testing", "jmeter", "bdd", "cucumber", "pact", "k6"],
        "tools":    ["selenium", "appium", "postman", "jira", "jenkins", "github actions", "browserstack"],
        "resources": [
            {"title": "Test Automation University", "type": "🏋️ Platform", "link": "https://testautomationu.applitools.com"},
            {"title": "Playwright Docs", "type": "📄 Docs", "link": "https://playwright.dev"},
            {"title": "ISTQB Cert", "type": "📜 Cert", "link": "https://www.istqb.org"},
        ]
    },
    "DotNet Developer": {
        "core":     ["c#", ".net core", "asp.net", "entity framework", "sql", "rest api", "linq", "azure", "visual studio", "git"],
        "advanced": ["blazor", "signalr", "microservices", "docker", "kubernetes", "azure devops", "grpc", "identity server"],
        "tools":    ["visual studio", "vs code", "azure devops", "sql server", "postman", "docker"],
        "resources": [
            {"title": "Microsoft Learn – .NET", "type": "🏋️ Platform", "link": "https://learn.microsoft.com/en-us/dotnet/"},
            {"title": "C# in Depth – Jon Skeet", "type": "📘 Book", "link": "https://www.manning.com/books/c-sharp-in-depth-fourth-edition"},
        ]
    },
    "Operations Manager": {
        "core":     ["operations management", "process improvement", "lean", "six sigma", "supply chain", "budgeting", "kpi tracking", "team leadership"],
        "advanced": ["erp systems", "vendor management", "logistics optimization", "capacity planning", "risk management", "total quality management"],
        "tools":    ["excel", "power bi", "sap", "oracle erp", "ms project", "tableau", "jira"],
        "resources": [
            {"title": "Lean Six Sigma Green Belt", "type": "📜 Cert", "link": "https://www.asq.org/cert/six-sigma-green-belt"},
            {"title": "APICS CPIM", "type": "📜 Cert", "link": "https://www.ascm.org/cpim/"},
        ]
    },
}

CATEGORY_MAP = {
    "Data Science": "Data Science",
    "Machine Learning": "Machine Learning",
    "Web Designing": "Web Designing",
    "Java Developer": "Java Developer",
    "Python Developer": "Python Developer",
    "DevOps Engineer": "DevOps Engineer",
    "Database": "Database",
    "Network Security Engineer": "Network Security Engineer",
    "Hadoop": "Hadoop",
    "ETL Developer": "ETL Developer",
    "Blockchain": "Blockchain",
    "Automation Testing": "Automation Testing",
    "DotNet Developer": "DotNet Developer",
    "Operations Manager": "Operations Manager",
}

# ─── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    import kagglehub

    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    csv_file = None
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.endswith(".csv"):
                csv_file = os.path.join(root, fn)
                break

    df = pd.read_csv(csv_file)
    df.dropna(subset=['Resume_str', 'Category'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_resume(text):
        text = str(text)
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(w) for w in tokens
                  if w not in stop_words and len(w) > 2]
        return ' '.join(tokens)

    df['cleaned_resume'] = df['Resume_str'].apply(clean_resume)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df['cleaned_resume'])

    lr_model = LogisticRegression(max_iter=1000, C=5, random_state=42)
    lr_model.fit(X_tfidf, df['label'])

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def truncate(text, max_words=150):
        return ' '.join(str(text).split()[:max_words])

    career_profiles = {}
    for category in df['Category'].unique():
        docs = df[df['Category'] == category]['cleaned_resume'].apply(truncate).tolist()
        embeddings = bert_model.encode(docs, batch_size=32, show_progress_bar=False)
        career_profiles[category] = embeddings.mean(axis=0)

    career_names = list(career_profiles.keys())
    career_vectors = np.array(list(career_profiles.values()))

    tfidf_full = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_full.fit(df['cleaned_resume'].tolist())
    tfidf_career_profiles = {}
    for category in df['Category'].unique():
        docs = df[df['Category'] == category]['cleaned_resume'].tolist()
        combined = ' '.join(docs)
        tfidf_career_profiles[category] = tfidf_full.transform([combined]).toarray()[0]

    return {
        "clean_resume": clean_resume,
        "tfidf": tfidf,
        "lr_model": lr_model,
        "le": le,
        "bert_model": bert_model,
        "career_names": career_names,
        "career_vectors": career_vectors,
        "tfidf_full": tfidf_full,
        "tfidf_career_names": list(tfidf_career_profiles.keys()),
        "tfidf_career_matrix": np.array(list(tfidf_career_profiles.values())),
        "truncate": truncate,
        "df": df,
    }


# ─── Helper functions ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    try:
        import pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        os.unlink(tmp_path)
        return text.strip()
    except ImportError:
        st.warning("⚠️ pdfplumber not installed. Run: `pip install pdfplumber`")
        return ""
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""


def extract_user_skills(resume_text):
    resume_lower = resume_text.lower()
    found_skills = set()
    all_skills = set()
    for career_data in CAREER_SKILLS.values():
        for key, skill_list in career_data.items():
            if isinstance(skill_list, list) and skill_list and isinstance(skill_list[0], str):
                all_skills.update(skill_list)
    for skill in all_skills:
        if skill.lower() in resume_lower:
            found_skills.add(skill)
    return found_skills


def get_skill_gap(user_skills, target_career):
    career_key = CATEGORY_MAP.get(target_career, target_career)
    if career_key not in CAREER_SKILLS:
        return {}, []
    career_data = CAREER_SKILLS[career_key]
    core_skills    = set(career_data.get("core", []))
    advanced_skills = set(career_data.get("advanced", []))
    tools          = set(career_data.get("tools", []))
    resources      = career_data.get("resources", [])
    user_lower     = {s.lower() for s in user_skills}

    return {
        "have_core":       sorted(core_skills & user_lower),
        "missing_core":    sorted(core_skills - user_lower),
        "have_advanced":   sorted(advanced_skills & user_lower),
        "missing_advanced": sorted(advanced_skills - user_lower),
        "have_tools":      sorted(tools & user_lower),
        "missing_tools":   sorted(tools - user_lower),
        "match_pct":       round(len(core_skills & user_lower) / max(len(core_skills), 1) * 100),
    }, resources


def recommend_career_bert(models, resume_text, top_n=5):
    cleaned = models["clean_resume"](resume_text)
    truncated = models["truncate"](cleaned)
    resume_vec = models["bert_model"].encode([truncated])
    sims = cosine_similarity(resume_vec, models["career_vectors"])[0]
    top_idx = sims.argsort()[::-1][:top_n]
    return [{"career": models["career_names"][i], "score": round(float(sims[i]) * 100, 1)} for i in top_idx]


def predict_career_tfidf(models, resume_text):
    cleaned = models["clean_resume"](resume_text)
    vec = models["tfidf"].transform([cleaned])
    label = models["lr_model"].predict(vec)[0]
    return models["le"].inverse_transform([label])[0]


def jd_skill_match(resume_text, jd_text):
    """Simple keyword overlap score between resume and JD."""
    resume_words = set(re.findall(r'\b[a-z][a-z0-9+#.]{1,}\b', resume_text.lower()))
    jd_words     = set(re.findall(r'\b[a-z][a-z0-9+#.]{1,}\b', jd_text.lower()))
    stop = {"and","the","for","with","you","are","our","have","will","to","of","in","a","an",
            "is","be","as","we","or","on","at","by","this","that","your","their","more","from"}
    jd_words -= stop
    if not jd_words:
        return 0, set(), set()
    matched = resume_words & jd_words
    missing = jd_words - resume_words
    score = round(len(matched) / len(jd_words) * 100, 1)
    return score, matched, missing


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem;padding-top:.5rem'>
      <div style='font-size:1.5rem;font-weight:800;letter-spacing:-.02em'>🧭 Career Advisor</div>
      <div style='font-size:.77rem;opacity:.55;margin-top:.25rem'>NLP-powered path finder</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-size:.68rem;opacity:.55;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.5rem'>Input method</div>", unsafe_allow_html=True)
    input_mode = st.radio("", ["📄 Upload Resume", "✏️ Paste Text", "🎮 Quick Demo"], label_visibility="collapsed")

    st.markdown("---")
    top_n = st.slider("Top N career matches", 3, 10, 5)

    st.markdown("---")
    enable_jd = st.checkbox("🔗 Enable JD Matcher", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem;opacity:.7;line-height:1.65'>
    <strong>How it works</strong><br>
    ① Upload your resume<br>
    ② BERT + TF-IDF analyze it<br>
    ③ Get career matches, skill gaps &amp; a roadmap
    </div>
    <div style='margin-top:1rem;font-size:.72rem;opacity:.45'>
    Model: all-MiniLM-L6-v2<br>
    Data: Kaggle Resume Dataset
    </div>
    """, unsafe_allow_html=True)


# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-badge'>✨ AI-Powered · NLP · BERT + TF-IDF</div>
  <h1>Your Career Path,<br><em>Decoded by AI</em></h1>
  <p>Upload your resume and get instant career recommendations,<br>skill gap analysis, and a personalized learning roadmap.</p>
</div>
""", unsafe_allow_html=True)


# ─── Input Section ──────────────────────────────────────────────────────────────
resume_text = ""

if input_mode == "📄 Upload Resume":
    st.markdown("<div class='section-label'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload your resume</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop your resume (PDF or TXT)",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded:
        if uploaded.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded.read())
            if not resume_text:
                # fallback raw decode
                uploaded.seek(0)
                resume_text = uploaded.read().decode("utf-8", errors="ignore")
        else:
            resume_text = uploaded.read().decode("utf-8", errors="ignore")

        if resume_text:
            col_prev, col_stat = st.columns([3, 1])
            with col_prev:
                with st.expander("👁️ Preview extracted text", expanded=False):
                    st.text(resume_text[:2000] + ("…" if len(resume_text) > 2000 else ""))
            with col_stat:
                wc = len(resume_text.split())
                st.markdown(f"""
                <div class='metric-tile' style='margin-top:.2rem'>
                  <div class='metric-num' style='font-size:1.5rem'>{wc}</div>
                  <div class='metric-label'>words extracted</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Could not extract text. Try the Paste Text option instead.")

elif input_mode == "✏️ Paste Text":
    st.markdown("<div class='section-label'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Paste your resume text</div>", unsafe_allow_html=True)
    resume_text = st.text_area(
        "Resume text",
        placeholder="Paste your full resume here — skills, experience, education, projects, certifications…",
        height=300,
        label_visibility="collapsed"
    )

else:  # Quick Demo
    st.markdown("<div class='section-label'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Choose a demo profile</div>", unsafe_allow_html=True)
    demo_options = {
        "🤖 ML / AI Engineer": """
Experienced ML engineer with 4 years building production machine learning systems.
Strong background in Python, scikit-learn, TensorFlow, PyTorch, BERT, transformers, NLP, computer vision.
Built data pipelines using pandas, numpy, spark; deployed models with Docker, Kubernetes, AWS SageMaker.
Proficient in git, MLflow, Weights & Biases, statistics, feature engineering, deep learning, A/B testing.
B.Tech Computer Science, IIT Bombay. Published research on NLP and reinforcement learning.
""",
        "🌐 Frontend Developer": """
Creative frontend developer with 4 years building responsive web applications.
Expert in HTML, CSS, JavaScript, React, TypeScript, Tailwind CSS, Figma, responsive design, accessibility.
Strong understanding of UX design, UI design, typography, color theory, and web performance.
Built 20+ production websites. Familiar with git, webpack, Vite, Vercel, and REST APIs.
""",
        "🗄️ Data Analyst": """
Data analyst with 3 years turning raw data into business insights.
Skilled in SQL, Python, pandas, numpy, Excel, Tableau, Power BI, statistics, data visualization.
Experience with A/B testing, ETL pipelines, PostgreSQL, data warehousing, and business reporting.
Strong communicator; translates technical findings for non-technical stakeholders.
""",
    }
    selected_demo = st.selectbox("Choose a demo profile:", list(demo_options.keys()))
    resume_text = demo_options[selected_demo]
    st.info("✅ Demo resume loaded. Click **Analyze My Resume** below.")


# ─── JD Matcher (optional) ──────────────────────────────────────────────────────
jd_text = ""
if enable_jd:
    st.markdown("---")
    st.markdown("<div class='section-label'>Optional</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Paste Job Description</div>", unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job description",
        placeholder="Paste the job description you're targeting — skills, requirements, responsibilities…",
        height=180,
        label_visibility="collapsed"
    )


# ─── Analyze button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_hint = st.columns([2, 5])
with col_btn:
    analyze = st.button("🔍 Analyze My Resume", use_container_width=True)
with col_hint:
    st.markdown("""
    <div style='padding-top:.6rem;font-size:.82rem;color:#94a3b8'>
    First run downloads models (~500 MB) and may take 60–90 s.
    Subsequent runs are instant.
    </div>
    """, unsafe_allow_html=True)


# ─── Results ────────────────────────────────────────────────────────────────────
if analyze and resume_text.strip():

    with st.spinner("⚙️ Loading AI models (first run ~60–90 s)…"):
        try:
            models = load_models()
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            st.info("Make sure all packages are installed. Run:\n```\npip install nltk sentence-transformers kagglehub scikit-learn pdfplumber pandas numpy\n```")
            st.stop()

    with st.spinner("🔬 Analyzing your resume…"):
        tfidf_pred  = predict_career_tfidf(models, resume_text)
        bert_recs   = recommend_career_bert(models, resume_text, top_n=top_n)
        user_skills = extract_user_skills(resume_text)
        top_career  = bert_recs[0]["career"] if bert_recs else tfidf_pred
        gap_data, resources = get_skill_gap(user_skills, top_career)

    st.markdown("---")

    # ── Quick stats ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    match_pct     = gap_data.get("match_pct", 0) if isinstance(gap_data, dict) else 0
    missing_count = len(gap_data.get("missing_core", [])) if isinstance(gap_data, dict) else 0

    with c1:
        st.markdown(f"""<div class='metric-tile'>
          <div class='metric-num'>{len(user_skills)}</div>
          <div class='metric-label'>Skills detected</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-tile'>
          <div class='metric-num'>{match_pct}%</div>
          <div class='metric-label'>Core skill match</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-tile'>
          <div class='metric-num'>{missing_count}</div>
          <div class='metric-label'>Skills to learn</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-tile'>
          <div class='metric-num'>{len(resources)}</div>
          <div class='metric-label'>Learning resources</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── JD Match box (if enabled) ─────────────────────────────────────────────
    if enable_jd and jd_text.strip():
        jd_score, jd_matched, jd_missing = jd_skill_match(resume_text, jd_text)
        st.markdown("<div class='section-label'>JD Match Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:white;border:1.5px solid #ede9ff;border-radius:16px;padding:1.3rem 1.5rem;margin-bottom:1.5rem'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem'>
            <div style='font-weight:700;font-size:1rem;color:#1a1a2e'>Resume ↔ Job Description Match</div>
            <div style='font-size:1.6rem;font-weight:800;color:#7c6ff7;font-family:"DM Serif Display",serif'>{jd_score}%</div>
          </div>
          <div class='jd-bar-bg'><div class='jd-bar' style='width:{min(jd_score,100)}%'></div></div>
          <div style='margin-top:.8rem;font-size:.82rem;color:#64748b'>
            <strong>{len(jd_matched)}</strong> keywords matched &nbsp;·&nbsp;
            <strong>{len(jd_missing)}</strong> JD keywords missing from resume
          </div>
        </div>
        """, unsafe_allow_html=True)
        if jd_missing:
            top_missing = sorted(list(jd_missing), key=len, reverse=True)[:20]
            missing_html = "".join([f"<span class='skill-missing'>{w}</span>" for w in top_missing])
            st.markdown(f"**Top keywords to add to your resume:**<br>{missing_html}", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Career Matches", "🔍 Skill Gap", "📚 Learning Roadmap", "📊 Deep Analysis"])

    # ── Tab 1: Career Matches ─────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.markdown("<div class='section-label'>BERT Semantic Similarity</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Your top career matches</div>", unsafe_allow_html=True)

            for i, rec in enumerate(bert_recs):
                top_class  = "top" if i == 0 else ""
                rank_label = "🏆 Best match" if i == 0 else f"#{i+1}"
                rank_class = "gold" if i == 0 else ""
                st.markdown(f"""
                <div class='career-card {top_class}'>
                  <div class='career-rank {rank_class}'>{rank_label}</div>
                  <div class='career-name'>{rec["career"]}</div>
                  <div style='font-size:.82rem;color:#64748b;margin-top:.1rem'>{rec["score"]}% semantic match</div>
                  <div class='sim-bar-bg'><div class='sim-bar' style='width:{min(rec["score"],100)}%'></div></div>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='section-label'>TF-IDF ML Classifier</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Direct prediction</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='career-card top' style='margin-top:.2rem'>
              <div style='font-size:.78rem;color:#7c6ff7;font-weight:700;margin-bottom:.4rem;letter-spacing:.04em'>LOGISTIC REGRESSION ON TF-IDF</div>
              <div style='font-size:1.5rem;font-weight:800;color:#1a1a2e'>{tfidf_pred}</div>
              <div style='font-size:.78rem;color:#64748b;margin-top:.3rem'>Keyword-based classification</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Skills detected in resume</div>", unsafe_allow_html=True)
            if user_skills:
                skills_html = "".join([f"<span class='skill-have'>{s}</span>" for s in sorted(user_skills)[:24]])
                st.markdown(f"<div style='line-height:2'>{skills_html}</div>", unsafe_allow_html=True)
                if len(user_skills) > 24:
                    st.caption(f"+ {len(user_skills)-24} more skills detected")
            else:
                st.caption("No specific skills matched. Try a more detailed resume.")

    # ── Tab 2: Skill Gap ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-label'>Skill Gap Analysis</div>", unsafe_allow_html=True)

        career_options = [r["career"] for r in bert_recs]
        selected_career = st.selectbox("Analyze skill gap for:", career_options, key="gap_select")
        gap_data2, resources2 = get_skill_gap(user_skills, selected_career)

        if isinstance(gap_data2, dict) and gap_data2:
            match2 = gap_data2.get("match_pct", 0)

            st.markdown(f"""
            <div style='margin:1rem 0 1.5rem;background:white;border:1.5px solid #ede9ff;border-radius:14px;padding:1.2rem 1.4rem'>
              <div style='display:flex;justify-content:space-between;font-size:.88rem;font-weight:600;margin-bottom:.5rem'>
                <span>Core skill match for <strong>{selected_career}</strong></span>
                <span style='color:#7c6ff7;font-size:1.1rem'>{match2}%</span>
              </div>
              <div style='height:12px;background:#f0eeff;border-radius:6px;overflow:hidden'>
                <div style='height:12px;width:{match2}%;background:linear-gradient(90deg,#7c6ff7,#a78bfa);border-radius:6px;transition:width .6s'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**🔵 Core Skills**")
                if gap_data2["have_core"]:
                    html = "".join([f"<span class='skill-have'>{s}</span>" for s in gap_data2["have_core"]])
                    st.markdown(f"<div style='line-height:2.1'>✅ You have:<br>{html}</div>", unsafe_allow_html=True)
                if gap_data2["missing_core"]:
                    html = "".join([f"<span class='skill-missing'>{s}</span>" for s in gap_data2["missing_core"]])
                    st.markdown(f"<div style='line-height:2.1;margin-top:.7rem'>❌ Missing:<br>{html}</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("**🟣 Advanced Skills**")
                if gap_data2["have_advanced"]:
                    html = "".join([f"<span class='skill-have'>{s}</span>" for s in gap_data2["have_advanced"]])
                    st.markdown(f"<div style='line-height:2.1'>✅ You have:<br>{html}</div>", unsafe_allow_html=True)
                if gap_data2["missing_advanced"]:
                    html = "".join([f"<span class='skill-learn'>{s}</span>" for s in gap_data2["missing_advanced"]])
                    st.markdown(f"<div style='line-height:2.1;margin-top:.7rem'>📖 To learn:<br>{html}</div>", unsafe_allow_html=True)

            with c3:
                st.markdown("**🟠 Tools & Platforms**")
                if gap_data2["have_tools"]:
                    html = "".join([f"<span class='skill-have'>{s}</span>" for s in gap_data2["have_tools"]])
                    st.markdown(f"<div style='line-height:2.1'>✅ You know:<br>{html}</div>", unsafe_allow_html=True)
                if gap_data2["missing_tools"]:
                    html = "".join([f"<span class='skill-learn'>{s}</span>" for s in gap_data2["missing_tools"]])
                    st.markdown(f"<div style='line-height:2.1;margin-top:.7rem'>🛠️ To add:<br>{html}</div>", unsafe_allow_html=True)
        else:
            st.info("No skill data available for this career in the knowledge base.")

    # ── Tab 3: Learning Roadmap ────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-label'>Personalized Roadmap</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>How to become a {top_career}</div>", unsafe_allow_html=True)

        gap_final, res_final = get_skill_gap(user_skills, top_career)

        if isinstance(gap_final, dict) and gap_final:
            priorities = gap_final.get("missing_core", [])[:5]
            adv_gaps   = gap_final.get("missing_advanced", [])[:5]

            # Phase 1
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:.8rem'>
              <span class='step-badge'>1</span>
              <div>
                <div style='font-weight:700;font-size:1rem;margin-bottom:.15rem'>Fill core skill gaps</div>
                <div style='font-size:.84rem;color:#64748b'>Must-haves for {top_career}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if priorities:
                html = "".join([f"<span class='skill-missing'>{s}</span>" for s in priorities])
                st.markdown(f"<div style='line-height:2.1;margin-left:2.8rem'>{html}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 You already have all core skills for this role!")

            st.markdown("<br>", unsafe_allow_html=True)

            # Phase 2
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:.8rem'>
              <span class='step-badge'>2</span>
              <div>
                <div style='font-weight:700;font-size:1rem;margin-bottom:.15rem'>Level up with advanced skills</div>
                <div style='font-size:.84rem;color:#64748b'>Stand out from other candidates</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if adv_gaps:
                html = "".join([f"<span class='skill-learn'>{s}</span>" for s in adv_gaps])
                st.markdown(f"<div style='line-height:2.1;margin-left:2.8rem'>{html}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 You're advanced in this field already!")

            st.markdown("<br>", unsafe_allow_html=True)

            # Phase 3
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:.8rem'>
              <span class='step-badge'>3</span>
              <div>
                <div style='font-weight:700;font-size:1rem;margin-bottom:.15rem'>Curated learning resources</div>
                <div style='font-size:.84rem;color:#64748b'>Handpicked for {top_career}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            for r in res_final:
                st.markdown(f"""
                <div class='resource-card'>
                  <div class='resource-title'>{r["title"]}</div>
                  <div class='resource-meta'>{r["type"]} &nbsp;·&nbsp; <a href='{r["link"]}' target='_blank' style='color:#7c6ff7;text-decoration:none'>Open resource ↗</a></div>
                </div>
                """, unsafe_allow_html=True)

            # Phase 4 — Next steps
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:.8rem'>
              <span class='step-badge'>4</span>
              <div>
                <div style='font-weight:700;font-size:1rem;margin-bottom:.15rem'>Practical next steps</div>
                <div style='font-size:.84rem;color:#64748b'>Apply your skills and get noticed</div>
              </div>
            </div>
            <div style='margin-left:2.8rem;font-size:.88rem;color:#475569;line-height:1.9'>
              🔨 Build 2–3 portfolio projects showcasing your skills<br>
              🐙 Contribute to open source GitHub repositories<br>
              📝 Update your LinkedIn with new skills & projects<br>
              🤝 Join communities: Discord, Reddit, Slack groups<br>
              💼 Apply with a tailored resume for each role
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Knowledge base entry not found for this career. More careers coming soon.")

    # ── Tab 4: Deep Analysis ───────────────────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-label'>Deep Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Full recommendation report</div>", unsafe_allow_html=True)

        st.markdown("**BERT similarity scores (all matches)**")
        bert_df = pd.DataFrame(bert_recs).rename(columns={"career": "Career Path", "score": "Semantic Match (%)"})
        st.dataframe(bert_df, use_container_width=True, hide_index=True)

        if user_skills:
            st.markdown("<br>**Detected skills inventory**", unsafe_allow_html=True)
            skill_df = pd.DataFrame({"Skill": sorted(user_skills)})
            n_cols = 3
            chunk = len(skill_df) // n_cols + 1
            skill_cols = st.columns(n_cols)
            for idx, col in enumerate(skill_cols):
                with col:
                    sub = skill_df.iloc[idx*chunk:(idx+1)*chunk]
                    if not sub.empty:
                        st.dataframe(sub, use_container_width=True, hide_index=True)

        st.markdown("<br>**Skill match % across top careers**", unsafe_allow_html=True)
        gap_table = []
        for rec in bert_recs[:5]:
            gd, _ = get_skill_gap(user_skills, rec["career"])
            if isinstance(gd, dict):
                gap_table.append({
                    "Career": rec["career"],
                    "Semantic Match (%)": rec["score"],
                    "Core Skill Match (%)": gd.get("match_pct", 0),
                    "Skills You Have": len(gd.get("have_core", [])) + len(gd.get("have_advanced", [])),
                    "Skills to Learn": len(gd.get("missing_core", [])) + len(gd.get("missing_advanced", [])),
                })
        if gap_table:
            st.dataframe(pd.DataFrame(gap_table), use_container_width=True, hide_index=True)

        # Download button
        st.markdown("<br>", unsafe_allow_html=True)
        report_lines = [
            "CAREER PATH ADVISOR — ANALYSIS REPORT",
            "=" * 50,
            f"Top career match: {top_career}",
            f"TF-IDF prediction: {tfidf_pred}",
            f"Skills detected: {len(user_skills)}",
            f"Core skill match: {match_pct}%",
            "",
            "TOP CAREER MATCHES (BERT):",
        ]
        for r in bert_recs:
            report_lines.append(f"  {r['career']}: {r['score']}%")
        report_lines += ["", "DETECTED SKILLS:"]
        report_lines += [f"  {s}" for s in sorted(user_skills)]
        report_lines += ["", "CORE SKILL GAPS:"]
        for s in gap_data.get("missing_core", []):
            report_lines.append(f"  ❌ {s}")

        st.download_button(
            label="⬇️ Download Report (.txt)",
            data="\n".join(report_lines),
            file_name="career_advisor_report.txt",
            mime="text/plain"
        )

elif analyze and not resume_text.strip():
    st.warning("⚠️ Please provide your resume text first.")

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center;padding:4rem 1rem 3rem;'>
      <div style='font-size:4rem;margin-bottom:1.2rem'>🧭</div>
      <div style='font-size:1.25rem;font-weight:700;color:#1a1a2e;font-family:"DM Serif Display",serif'>Ready when you are</div>
      <div style='font-size:.92rem;margin-top:.5rem;color:#94a3b8;max-width:400px;margin-left:auto;margin-right:auto'>
        Upload your resume or paste your text,<br>then click <strong style='color:#7c6ff7'>Analyze My Resume</strong>
      </div>
      <div style='margin-top:2rem;display:flex;gap:1.2rem;justify-content:center;flex-wrap:wrap'>
        <div style='background:#f8f7ff;border:1.5px solid #ede9ff;border-radius:12px;padding:.9rem 1.4rem;font-size:.84rem;color:#475569'>
          🤖 BERT Semantic Similarity
        </div>
        <div style='background:#f8f7ff;border:1.5px solid #ede9ff;border-radius:12px;padding:.9rem 1.4rem;font-size:.84rem;color:#475569'>
          📊 TF-IDF ML Classification
        </div>
        <div style='background:#f8f7ff;border:1.5px solid #ede9ff;border-radius:12px;padding:.9rem 1.4rem;font-size:.84rem;color:#475569'>
          🎯 Skill Gap Analysis
        </div>
        <div style='background:#f8f7ff;border:1.5px solid #ede9ff;border-radius:12px;padding:.9rem 1.4rem;font-size:.84rem;color:#475569'>
          🗺️ Learning Roadmap
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
