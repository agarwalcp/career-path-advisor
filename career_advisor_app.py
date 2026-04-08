"""
CareerLens AI — Fixed & Enhanced Edition
ROOT FIX: CATEGORY_MAP now uses the EXACT names from the Kaggle dataset
(ACCOUNTANT, INFORMATION-TECHNOLOGY, ENGINEERING, etc.)
All 24 dataset categories are mapped to skill knowledge base.
Dark Glassmorphism UI — Syne + Space Grotesk + JetBrains Mono
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, os, tempfile
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CareerLens AI", page_icon="🔭", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════
#  FULL CSS — Dark Glassmorphism
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{background:#040812!important;color:#e2e8f0!important;font-family:'Space Grotesk',sans-serif!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse 80% 50% at 15% 10%,rgba(99,102,241,.13) 0%,transparent 60%),
             radial-gradient(ellipse 60% 45% at 85% 85%,rgba(16,185,129,.09) 0%,transparent 60%),
             radial-gradient(ellipse 55% 65% at 50% 50%,rgba(139,92,246,.06) 0%,transparent 70%),
             #040812;pointer-events:none;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#0f172a;}::-webkit-scrollbar-thumb{background:#3730a3;border-radius:2px;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px!important;}

/* ── HERO ── */
.hero-wrap{position:relative;border-radius:24px;overflow:hidden;margin-bottom:2.5rem;padding:3.5rem 3rem 3rem;
  background:linear-gradient(135deg,rgba(13,18,38,.97) 0%,rgba(28,24,65,.93) 55%,rgba(13,18,38,.97) 100%);
  border:1px solid rgba(99,102,241,.22);
  box-shadow:0 0 0 1px rgba(99,102,241,.07),0 30px 60px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.04);}
.hero-wrap::before{content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 40px,rgba(99,102,241,.022) 40px,rgba(99,102,241,.022) 41px),
             repeating-linear-gradient(90deg,transparent,transparent 40px,rgba(99,102,241,.022) 40px,rgba(99,102,241,.022) 41px);
  pointer-events:none;}
.hero-eye{display:inline-flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;
  font-size:.68rem;font-weight:500;letter-spacing:.15em;text-transform:uppercase;
  color:#10b981;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.22);
  padding:5px 14px;border-radius:99px;margin-bottom:1.2rem;}
.hero-eye::before{content:'';width:6px;height:6px;border-radius:50%;background:#10b981;
  box-shadow:0 0 8px #10b981;animation:pdot 2s infinite;}
@keyframes pdot{0%,100%{opacity:1;box-shadow:0 0 8px #10b981;}50%{opacity:.4;box-shadow:0 0 18px #10b981;}}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(2.4rem,5vw,3.8rem);font-weight:800;
  line-height:1.05;letter-spacing:-.03em;color:#f8fafc;margin-bottom:.8rem;position:relative;z-index:1;}
.hero-title span{background:linear-gradient(90deg,#818cf8,#a78bfa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:1rem;font-weight:300;color:rgba(148,163,184,.82);max-width:540px;line-height:1.75;margin-bottom:2rem;}
.hero-pills{display:flex;flex-wrap:wrap;gap:.45rem;}
.hero-pill{font-size:.7rem;font-weight:500;color:rgba(165,180,252,.88);
  background:rgba(99,102,241,.09);border:1px solid rgba(99,102,241,.18);padding:4px 11px;border-radius:99px;}
.hero-stats{position:absolute;top:2.5rem;right:2.5rem;display:flex;gap:1.8rem;}
.hs-num{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#f8fafc;line-height:1;}
.hs-lbl{font-size:.62rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:rgba(148,163,184,.45);margin-top:.2rem;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:rgba(5,7,16,.98)!important;border-right:1px solid rgba(99,102,241,.12)!important;}
[data-testid="stSidebar"]>div{padding-top:1.5rem!important;}
.sb-logo{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#f8fafc!important;}
.sb-logo span{color:#818cf8;}
.sb-ver{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:rgba(148,163,184,.3)!important;margin-top:.2rem;letter-spacing:.08em;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] .stMarkdown{color:rgba(226,232,240,.72)!important;}
[data-testid="stSidebar"] hr{border-color:rgba(99,102,241,.12)!important;}
.sb-sec{font-family:'JetBrains Mono',monospace;font-size:.58rem;letter-spacing:.15em;text-transform:uppercase;color:rgba(99,102,241,.6)!important;margin-bottom:.4rem;}

/* ── SECTION LABELS ── */
.sec-eye{font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#818cf8;margin-bottom:.35rem;}
.sec-head{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:#f1f5f9;margin-bottom:1.1rem;letter-spacing:-.02em;}

/* ── METRIC TILES ── */
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;}
.metric-tile{background:rgba(13,18,38,.85);border:1px solid rgba(99,102,241,.14);border-radius:16px;
  padding:1.3rem 1rem;text-align:center;position:relative;overflow:hidden;transition:all .3s;}
.metric-tile::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#6366f1,#8b5cf6,#34d399);transform:scaleX(0);transition:transform .3s;}
.metric-tile:hover::after{transform:scaleX(1);}
.metric-tile:hover{border-color:rgba(99,102,241,.35);box-shadow:0 8px 24px rgba(99,102,241,.12);transform:translateY(-2px);}
.mt-icon{font-size:1.25rem;margin-bottom:.3rem;}
.mt-num{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
  background:linear-gradient(135deg,#818cf8,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;}
.mt-lbl{font-size:.68rem;font-weight:500;letter-spacing:.07em;text-transform:uppercase;color:rgba(148,163,184,.48);margin-top:.35rem;}

/* ── MATCH CARDS ── */
.mc{background:rgba(13,18,38,.82);border:1px solid rgba(99,102,241,.13);border-radius:15px;
  padding:1.15rem 1.35rem;margin-bottom:.7rem;position:relative;overflow:hidden;transition:all .28s;}
.mc::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,#6366f1,#8b5cf6);transform:scaleY(0);transition:transform .28s;border-radius:3px 0 0 3px;}
.mc:hover{border-color:rgba(99,102,241,.38);transform:translateX(4px);box-shadow:0 6px 20px rgba(99,102,241,.1);}
.mc:hover::before{transform:scaleY(1);}
.mc.top{border-color:rgba(129,140,248,.38);background:rgba(28,24,65,.55);}
.mc.top::before{transform:scaleY(1);}
.mc-rank{position:absolute;top:.95rem;right:1.1rem;font-family:'JetBrains Mono',monospace;
  font-size:.62rem;font-weight:600;letter-spacing:.1em;color:#818cf8;
  background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.18);padding:3px 9px;border-radius:99px;}
.mc-rank.gld{color:#fbbf24;background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.22);}
.mc-name{font-family:'Syne',sans-serif;font-size:1.03rem;font-weight:700;color:#f1f5f9;margin-bottom:.18rem;}
.mc-score{font-size:.78rem;color:rgba(148,163,184,.55);margin-bottom:.55rem;}
.bar-bg{height:5px;background:rgba(99,102,241,.1);border-radius:3px;overflow:hidden;}
.bar-fg{height:5px;border-radius:3px;background:linear-gradient(90deg,#6366f1,#8b5cf6,#a78bfa);}

/* ── PRED BOX ── */
.pred-box{background:rgba(28,24,65,.5);border:1px solid rgba(129,140,248,.28);border-radius:15px;
  padding:1.4rem 1.5rem;position:relative;overflow:hidden;}
.pred-box::after{content:'';position:absolute;top:-30px;right:-30px;width:100px;height:100px;
  border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,.12),transparent 70%);}
.pred-lbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:600;
  letter-spacing:.15em;text-transform:uppercase;color:#818cf8;margin-bottom:.45rem;}
.pred-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#f8fafc;letter-spacing:-.02em;}

/* ── PILLS ── */
.ph{display:inline-block;background:rgba(16,185,129,.09);color:#34d399;border:1px solid rgba(16,185,129,.22);
  border-radius:99px;padding:3px 11px;font-size:.76rem;font-weight:500;margin:3px;transition:all .2s;}
.ph:hover{background:rgba(16,185,129,.16);}
.pm{display:inline-block;background:rgba(239,68,68,.08);color:#f87171;border:1px solid rgba(239,68,68,.18);
  border-radius:99px;padding:3px 11px;font-size:.76rem;font-weight:500;margin:3px;transition:all .2s;}
.pl{display:inline-block;background:rgba(251,191,36,.07);color:#fbbf24;border:1px solid rgba(251,191,36,.18);
  border-radius:99px;padding:3px 11px;font-size:.76rem;font-weight:500;margin:3px;transition:all .2s;}
.pn{display:inline-block;background:rgba(99,102,241,.08);color:rgba(165,180,252,.8);border:1px solid rgba(99,102,241,.15);
  border-radius:99px;padding:3px 11px;font-size:.76rem;font-weight:500;margin:3px;}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{background:rgba(13,18,38,.75)!important;border-radius:12px!important;
  padding:4px!important;border:1px solid rgba(99,102,241,.14)!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:9px!important;
  color:rgba(148,163,184,.62)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:500!important;font-size:.86rem!important;padding:.48rem 1.15rem!important;border:none!important;transition:all .2s!important;}
.stTabs [data-baseweb="tab"]:hover{color:#f1f5f9!important;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,.22)!important;color:#a5b4fc!important;
  box-shadow:0 0 12px rgba(99,102,241,.12)!important;}
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}

/* ── BUTTON ── */
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;color:#fff!important;
  border:none!important;border-radius:12px!important;font-family:'Syne',sans-serif!important;
  font-weight:700!important;font-size:.95rem!important;letter-spacing:.02em!important;
  padding:.68rem 2.2rem!important;box-shadow:0 4px 20px rgba(79,70,229,.42)!important;transition:all .28s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(79,70,229,.55)!important;}

/* ── UPLOAD ── */
[data-testid="stFileUploadDropzone"]{background:rgba(13,18,38,.8)!important;
  border:2px dashed rgba(99,102,241,.28)!important;border-radius:15px!important;transition:border-color .3s!important;}
[data-testid="stFileUploadDropzone"]:hover{border-color:rgba(99,102,241,.55)!important;}
[data-testid="stFileUploadDropzone"] *{color:rgba(148,163,184,.72)!important;}

/* ── INPUTS ── */
.stTextArea textarea,.stTextInput input{background:rgba(13,18,38,.82)!important;
  border:1px solid rgba(99,102,241,.18)!important;border-radius:11px!important;color:#e2e8f0!important;
  font-family:'Space Grotesk',sans-serif!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:rgba(99,102,241,.48)!important;
  box-shadow:0 0 0 3px rgba(99,102,241,.08)!important;}
.stSelectbox>div>div{background:rgba(13,18,38,.82)!important;border:1px solid rgba(99,102,241,.18)!important;
  border-radius:10px!important;color:#e2e8f0!important;}

/* ── EXPANDER ── */
.streamlit-expanderHeader{background:rgba(13,18,38,.72)!important;border:1px solid rgba(99,102,241,.13)!important;
  border-radius:10px!important;color:rgba(165,180,252,.88)!important;font-family:'Space Grotesk',sans-serif!important;font-weight:500!important;}
.streamlit-expanderContent{background:rgba(6,8,18,.65)!important;border:1px solid rgba(99,102,241,.09)!important;
  border-top:none!important;border-radius:0 0 10px 10px!important;}

/* ── ALERTS ── */
.stAlert{background:rgba(13,18,38,.82)!important;border-radius:12px!important;border:1px solid rgba(99,102,241,.18)!important;color:#e2e8f0!important;}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{border:1px solid rgba(99,102,241,.13)!important;border-radius:12px!important;overflow:hidden;}
[data-testid="stDataFrame"] th{background:rgba(28,24,65,.75)!important;color:#a5b4fc!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.7rem!important;letter-spacing:.06em!important;}
[data-testid="stDataFrame"] td{color:#cbd5e1!important;}

/* ── SPINNER ── */
.stSpinner>div{border-top-color:#6366f1!important;}

/* ── GAP OVERVIEW ── */
.gap-box{background:rgba(13,18,38,.82);border:1px solid rgba(99,102,241,.16);border-radius:15px;padding:1.3rem 1.5rem;margin-bottom:1.8rem;}
.gap-row{display:flex;justify-content:space-between;font-size:.82rem;font-weight:600;margin-bottom:.3rem;color:#e2e8f0;}
.gap-pct{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:#818cf8;}
.gap-track{height:11px;background:rgba(99,102,241,.09);border-radius:6px;overflow:hidden;margin:.4rem 0;}
.gap-fill{height:11px;border-radius:6px;background:linear-gradient(90deg,#4f46e5,#7c3aed,#34d399);}

/* ── SKILL COLUMN HEADER ── */
.skill-col-head{font-weight:700;font-size:.88rem;color:#f1f5f9;margin-bottom:.6rem;padding:.5rem .7rem;
  background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.12);border-radius:9px;}
.skill-section-lbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:700;
  letter-spacing:.12em;text-transform:uppercase;margin:.55rem 0 .3rem;}

/* ── STEP ROWS ── */
.step-row{display:flex;align-items:flex-start;gap:.85rem;margin-bottom:1.1rem;}
.step-n{flex-shrink:0;width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;
  font-family:'Syne',sans-serif;font-size:.78rem;font-weight:800;color:white;box-shadow:0 0 14px rgba(99,102,241,.38);}
.step-t{font-family:'Syne',sans-serif;font-size:.93rem;font-weight:700;color:#f1f5f9;margin-bottom:.1rem;}
.step-s{font-size:.78rem;color:rgba(148,163,184,.52);}

/* ── RESOURCE CARDS ── */
.res-card{background:rgba(13,18,38,.82);border:1px solid rgba(99,102,241,.1);border-left:3px solid #6366f1;
  border-radius:0 13px 13px 0;padding:.88rem 1.1rem;margin-bottom:.5rem;transition:all .22s;}
.res-card:hover{border-left-color:#34d399;box-shadow:0 4px 14px rgba(52,211,153,.07);transform:translateX(3px);}
.res-t{font-weight:600;font-size:.86rem;color:#f1f5f9;}
.res-m{font-size:.72rem;color:rgba(148,163,184,.48);margin-top:3px;}

/* ── NEXT STEPS ── */
.ns-item{display:flex;align-items:center;gap:.75rem;background:rgba(13,18,38,.72);
  border:1px solid rgba(99,102,241,.09);border-radius:10px;padding:.68rem 1rem;
  margin-bottom:.42rem;font-size:.83rem;color:#cbd5e1;transition:all .2s;}
.ns-item:hover{border-color:rgba(99,102,241,.28);color:#f1f5f9;}

/* ── JD BOX ── */
.jd-box{background:rgba(13,18,38,.82);border:1px solid rgba(99,102,241,.18);
  border-radius:17px;padding:1.5rem 1.6rem;margin-bottom:1.5rem;}
.jd-track{height:9px;background:rgba(99,102,241,.09);border-radius:5px;overflow:hidden;margin:.5rem 0;}
.jd-fill{height:9px;border-radius:5px;background:linear-gradient(90deg,#4f46e5,#7c3aed,#34d399);}

/* ── LANDING ── */
.land-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.9rem;max-width:660px;margin:0 auto;}
.land-f{background:rgba(13,18,38,.72);border:1px solid rgba(99,102,241,.11);border-radius:15px;
  padding:1.25rem 1.35rem;transition:all .28s;}
.land-f:hover{border-color:rgba(99,102,241,.32);box-shadow:0 6px 20px rgba(99,102,241,.09);transform:translateY(-2px);}
.lf-icon{font-size:1.5rem;margin-bottom:.4rem;}
.lf-t{font-family:'Syne',sans-serif;font-size:.92rem;font-weight:700;color:#f1f5f9;margin-bottom:.2rem;}
.lf-d{font-size:.76rem;color:rgba(148,163,184,.52);line-height:1.55;}

/* ── WC BADGE ── */
.wc-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,.07);
  border:1px solid rgba(16,185,129,.18);border-radius:99px;padding:4px 13px;
  font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:500;color:#34d399;margin-top:.55rem;}

/* ── DOWNLOAD BTN ── */
[data-testid="stDownloadButton"] button{background:rgba(13,18,38,.82)!important;
  border:1px solid rgba(99,102,241,.28)!important;color:#818cf8!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:500!important;transition:all .2s!important;}
[data-testid="stDownloadButton"] button:hover{background:rgba(99,102,241,.14)!important;
  border-color:rgba(99,102,241,.48)!important;color:#a5b4fc!important;}

hr{border:none!important;border-top:1px solid rgba(99,102,241,.1)!important;margin:1.8rem 0!important;}

/* ── CATEGORY WARNING BOX ── */
.no-data-box{background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.2);border-radius:12px;
  padding:1rem 1.2rem;font-size:.84rem;color:rgba(251,191,36,.85);line-height:1.6;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — keyed by EXACT Kaggle category names
#  (ALL CAPS with hyphens, matching df['Category'].unique())
# ══════════════════════════════════════════════════════════
CAREER_SKILLS = {
    # ── TECH CAREERS ──────────────────────────────────────
    "INFORMATION-TECHNOLOGY": {
        "core":     ["python","machine learning","scikit-learn","tensorflow","pytorch","numpy","pandas","sql","linux","networking","git","docker"],
        "advanced": ["deep learning","kubernetes","microservices","cloud computing","mlops","transformers","nlp","computer vision","devops","spark"],
        "tools":    ["aws","gcp","azure","jupyter","vscode","github","postman","jira","confluence"],
        "resources": [
            {"title":"CS50 – Introduction to Computer Science","type":"🎓 Course","link":"https://cs50.harvard.edu"},
            {"title":"AWS Certified Solutions Architect","type":"📜 Cert","link":"https://aws.amazon.com/certification/"},
            {"title":"Linux Foundation Training","type":"🏋️ Platform","link":"https://training.linuxfoundation.org"},
            {"title":"The Pragmatic Programmer","type":"📘 Book","link":"https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/"},
        ]
    },
    "ENGINEERING": {
        "core":     ["python","matlab","autocad","solidworks","mechanical design","electrical circuits","project management","statistics","simulation","testing"],
        "advanced": ["finite element analysis","cad/cam","plc programming","scada","embedded systems","robotics","six sigma","lean manufacturing"],
        "tools":    ["matlab","autocad","solidworks","ansys","catia","labview","git","jira","ms project"],
        "resources": [
            {"title":"MIT OpenCourseWare – Engineering","type":"🎓 Course","link":"https://ocw.mit.edu"},
            {"title":"Coursera Engineering Specializations","type":"🎓 Course","link":"https://www.coursera.org/browse/engineering"},
            {"title":"IEEE Spectrum","type":"📰 Publication","link":"https://spectrum.ieee.org"},
        ]
    },
    "DESIGNER": {
        "core":     ["figma","adobe xd","photoshop","illustrator","ui design","ux design","typography","color theory","responsive design","wireframing"],
        "advanced": ["motion design","after effects","3d modeling","blender","design systems","accessibility","user research","prototyping","animation"],
        "tools":    ["figma","adobe creative suite","sketch","invision","zeplin","miro","notion","framer"],
        "resources": [
            {"title":"Google UX Design Certificate","type":"📜 Cert","link":"https://grow.google/certificates/ux-design/"},
            {"title":"Refactoring UI","type":"📘 Book","link":"https://www.refactoringui.com"},
            {"title":"Nielsen Norman Group","type":"📰 Blog","link":"https://www.nngroup.com/articles/"},
        ]
    },
    "DIGITAL-MEDIA": {
        "core":     ["social media","content creation","seo","google analytics","copywriting","video editing","email marketing","adobe premiere","canva","wordpress"],
        "advanced": ["paid advertising","facebook ads","google ads","marketing automation","hubspot","a/b testing","cro","influencer marketing","podcast production"],
        "tools":    ["hootsuite","buffer","canva","adobe premiere","final cut pro","google analytics","semrush","mailchimp"],
        "resources": [
            {"title":"Google Digital Marketing Certificate","type":"📜 Cert","link":"https://grow.google/certificates/digital-marketing-ecommerce/"},
            {"title":"HubSpot Academy","type":"🏋️ Platform","link":"https://academy.hubspot.com"},
            {"title":"Moz SEO Learning Center","type":"🏋️ Platform","link":"https://moz.com/learn/seo"},
        ]
    },
    "CONSULTANT": {
        "core":     ["business analysis","project management","stakeholder management","powerpoint","excel","data analysis","problem solving","communication","consulting","strategy"],
        "advanced": ["change management","agile","six sigma","financial modeling","sql","tableau","power bi","process improvement","erp","risk management"],
        "tools":    ["ms office","excel","powerpoint","tableau","power bi","salesforce","jira","slack"],
        "resources": [
            {"title":"McKinsey Case Interview Prep","type":"🎓 Course","link":"https://www.mckinsey.com/careers/interviewing"},
            {"title":"Coursera Business Analytics","type":"🎓 Course","link":"https://www.coursera.org/specializations/business-analytics"},
        ]
    },
    # ── FINANCE & BUSINESS ────────────────────────────────
    "ACCOUNTANT": {
        "core":     ["accounting","financial statements","excel","quickbooks","tax","auditing","bookkeeping","balance sheet","payroll","gaap"],
        "advanced": ["financial modeling","erp","sap","oracle financials","sql","power bi","ifrs","forensic accounting","budgeting","variance analysis"],
        "tools":    ["quickbooks","sap","oracle","excel","xero","tally","ms dynamics","power bi"],
        "resources": [
            {"title":"CPA Exam Study Guide","type":"📜 Cert","link":"https://www.aicpa-cima.com/certifications/certified-public-accountant"},
            {"title":"AccountingCoach","type":"🏋️ Platform","link":"https://www.accountingcoach.com"},
            {"title":"Coursera Financial Accounting","type":"🎓 Course","link":"https://www.coursera.org/learn/wharton-accounting"},
        ]
    },
    "FINANCE": {
        "core":     ["financial analysis","excel","financial modeling","bloomberg","investment","portfolio management","risk management","accounting","valuation","sql"],
        "advanced": ["derivatives","fixed income","equity research","cfa","python","vba","machine learning","alternative investments","hedge fund","private equity"],
        "tools":    ["bloomberg","excel","python","r","sql","tableau","power bi","reuters eikon"],
        "resources": [
            {"title":"CFA Institute","type":"📜 Cert","link":"https://www.cfainstitute.org"},
            {"title":"Investopedia Academy","type":"🏋️ Platform","link":"https://academy.investopedia.com"},
            {"title":"Wall Street Prep","type":"🎓 Course","link":"https://www.wallstreetprep.com"},
        ]
    },
    "BANKING": {
        "core":     ["financial analysis","credit analysis","banking","excel","risk management","lending","compliance","customer service","kyc","anti-money laundering"],
        "advanced": ["financial modeling","python","sql","bloomberg","investment banking","treasury","derivatives","regulatory reporting","stress testing"],
        "tools":    ["bloomberg","excel","sql","python","temenos","sap","ms dynamics"],
        "resources": [
            {"title":"CFA Institute","type":"📜 Cert","link":"https://www.cfainstitute.org"},
            {"title":"FMVA – Financial Modeling","type":"📜 Cert","link":"https://corporatefinanceinstitute.com/certifications/financial-modeling-valuation-analyst-fmva/"},
        ]
    },
    # ── HR & BUSINESS ─────────────────────────────────────
    "HR": {
        "core":     ["recruitment","employee relations","payroll","performance management","onboarding","hris","labor law","compensation","training","talent acquisition"],
        "advanced": ["hr analytics","organizational development","succession planning","diversity & inclusion","workday","sap hr","leadership development","employer branding"],
        "tools":    ["workday","sap successfactors","bamboohr","linkedin recruiter","microsoft teams","excel","tableau"],
        "resources": [
            {"title":"SHRM Certification","type":"📜 Cert","link":"https://www.shrm.org/certification"},
            {"title":"LinkedIn Learning HR","type":"🏋️ Platform","link":"https://www.linkedin.com/learning/topics/hr"},
            {"title":"Josh Bersin Academy","type":"🎓 Course","link":"https://joshbersin.com/academy/"},
        ]
    },
    "BUSINESS-DEVELOPMENT": {
        "core":     ["sales","business development","crm","negotiation","lead generation","market research","excel","powerpoint","networking","cold calling"],
        "advanced": ["salesforce","hubspot","account management","partnership development","financial modeling","sql","data analysis","product strategy","gtm strategy"],
        "tools":    ["salesforce","hubspot","linkedin sales navigator","excel","powerpoint","zoom","slack"],
        "resources": [
            {"title":"Salesforce Trailhead","type":"🏋️ Platform","link":"https://trailhead.salesforce.com"},
            {"title":"HubSpot Sales Certification","type":"📜 Cert","link":"https://academy.hubspot.com/courses/sales-training"},
            {"title":"The Challenger Sale","type":"📘 Book","link":"https://www.challengerinc.com/the-challenger-sale-book/"},
        ]
    },
    "SALES": {
        "core":     ["sales","crm","cold calling","lead generation","negotiation","customer relationship","product knowledge","excel","communication","closing"],
        "advanced": ["salesforce","hubspot","account executive","enterprise sales","saas sales","pipeline management","territory management","sales analytics"],
        "tools":    ["salesforce","hubspot","outreach","linkedin","zoom","excel","slack","gong"],
        "resources": [
            {"title":"Salesforce Trailhead","type":"🏋️ Platform","link":"https://trailhead.salesforce.com"},
            {"title":"SPIN Selling","type":"📘 Book","link":"https://www.amazon.com/SPIN-Selling-Neil-Rackham/dp/0070511136"},
        ]
    },
    # ── HEALTHCARE & WELLNESS ──────────────────────────────
    "HEALTHCARE": {
        "core":     ["patient care","medical terminology","emr","hipaa","clinical documentation","nursing","diagnosis","treatment planning","medication management","vital signs"],
        "advanced": ["telemedicine","healthcare analytics","hl7","fhir","medical coding","icd-10","population health","care coordination","clinical research"],
        "tools":    ["epic","cerner","meditech","athenahealth","ms office","tableau"],
        "resources": [
            {"title":"Coursera Healthcare Specializations","type":"🎓 Course","link":"https://www.coursera.org/browse/health"},
            {"title":"HealthIT.gov Training","type":"🏋️ Platform","link":"https://www.healthit.gov/topic/onc-hitech-programs/workforce-development-programs"},
        ]
    },
    "FITNESS": {
        "core":     ["personal training","exercise science","nutrition","program design","anatomy","client assessment","cpr","strength training","cardio","coaching"],
        "advanced": ["sports performance","corrective exercise","group fitness","sports nutrition","rehabilitation","wearable tech","online coaching","business development"],
        "tools":    ["trainerize","mindbody","myfitnesspal","exercise.com","zoom","canva"],
        "resources": [
            {"title":"NASM Certified Personal Trainer","type":"📜 Cert","link":"https://www.nasm.org/certified-personal-trainer"},
            {"title":"ACE Fitness","type":"📜 Cert","link":"https://www.acefitness.org/fitness-certifications/"},
        ]
    },
    # ── EDUCATION ─────────────────────────────────────────
    "TEACHER": {
        "core":     ["lesson planning","curriculum development","classroom management","assessment","differentiated instruction","communication","google classroom","microsoft teams","student engagement","special education"],
        "advanced": ["instructional design","edtech","data-driven instruction","project-based learning","sel","grant writing","stem","blended learning","lms administration"],
        "tools":    ["google classroom","canvas","blackboard","microsoft teams","kahoot","nearpod","zoom","powerpoint"],
        "resources": [
            {"title":"Google for Education Training","type":"🏋️ Platform","link":"https://edu.google.com/intl/ALL_us/for-educators/certification/"},
            {"title":"Coursera Education Courses","type":"🎓 Course","link":"https://www.coursera.org/browse/social-sciences/education"},
        ]
    },
    "ADVOCATE": {
        "core":     ["legal research","case management","client counseling","litigation","contract drafting","negotiation","compliance","legal writing","court procedures","legislation"],
        "advanced": ["corporate law","intellectual property","mergers & acquisitions","arbitration","regulatory affairs","legal analytics","e-discovery","blockchain law"],
        "tools":    ["westlaw","lexisnexis","clio","ms office","docusign","relativity"],
        "resources": [
            {"title":"Bar Exam Prep – Themis","type":"📜 Cert","link":"https://www.themisbar.com"},
            {"title":"Coursera Law Specializations","type":"🎓 Course","link":"https://www.coursera.org/browse/social-sciences/law"},
        ]
    },
    # ── SKILLED TRADES & INDUSTRY ─────────────────────────
    "CHEF": {
        "core":     ["culinary arts","food safety","haccp","menu planning","kitchen management","food cost control","recipe development","team management","catering","inventory"],
        "advanced": ["molecular gastronomy","restaurant management","food styling","food photography","nutritional planning","global cuisines","franchise management","catering operations"],
        "tools":    ["restaurant365","toast pos","square","ms office","canva"],
        "resources": [
            {"title":"Culinary Institute of America","type":"🎓 Course","link":"https://www.ciachef.edu"},
            {"title":"ServSafe Certification","type":"📜 Cert","link":"https://www.servsafe.com"},
        ]
    },
    "CONSTRUCTION": {
        "core":     ["project management","blueprint reading","autocad","cost estimation","safety management","osha","scheduling","construction management","procurement","building codes"],
        "advanced": ["bim","revit","lean construction","green building","leed","primavera","ms project","risk management","contract management"],
        "tools":    ["autocad","revit","ms project","primavera","procore","bluebeam","excel"],
        "resources": [
            {"title":"PMP Certification","type":"📜 Cert","link":"https://www.pmi.org/certifications/project-management-pmp"},
            {"title":"Autodesk Training","type":"🏋️ Platform","link":"https://www.autodesk.com/training"},
        ]
    },
    "AUTOMOBILE": {
        "core":     ["automotive repair","diagnostics","obd2","engine repair","electrical systems","transmission","brake systems","customer service","parts management","safety inspection"],
        "advanced": ["ev/hybrid systems","adas","automotive software","dealership management","warranty claims","fleet management","automotive engineering","connected vehicles"],
        "tools":    ["mitchell1","alldata","identifix","reynolds and reynolds","excel","dealer management system"],
        "resources": [
            {"title":"ASE Certification","type":"📜 Cert","link":"https://www.ase.com"},
            {"title":"Automotive Training Center","type":"🏋️ Platform","link":"https://www.autotrainingcentre.com"},
        ]
    },
    "AVIATION": {
        "core":     ["aircraft maintenance","faa regulations","flight operations","navigation","air traffic","safety protocols","avionics","meteorology","aircraft systems","weight & balance"],
        "advanced": ["instrument rating","multi-engine","airline transport pilot","cabin crew","flight dispatch","aviation safety management","drone operations","mro management"],
        "tools":    ["foreflight","garmin avionics","jeppesen","ms office","faa databases"],
        "resources": [
            {"title":"FAA Training & Testing","type":"🎓 Course","link":"https://www.faa.gov/training_testing"},
            {"title":"IATA Training","type":"🎓 Course","link":"https://www.iata.org/en/training/"},
        ]
    },
    "AGRICULTURE": {
        "core":     ["crop management","soil science","irrigation","pest management","agronomy","livestock management","food safety","agricultural equipment","farm management","sustainability"],
        "advanced": ["precision agriculture","gis/remote sensing","agricultural data analytics","drone technology","supply chain management","organic farming","agri-business","climate science"],
        "tools":    ["trimble","john deere precision ag","arcgis","excel","farm management software"],
        "resources": [
            {"title":"Coursera Agriculture Courses","type":"🎓 Course","link":"https://www.coursera.org/courses?query=agriculture"},
            {"title":"FAO e-Learning","type":"🏋️ Platform","link":"https://elearning.fao.org"},
        ]
    },
    "APPAREL": {
        "core":     ["fashion design","textile knowledge","pattern making","visual merchandising","retail management","trend forecasting","inventory management","buyer","sourcing","quality control"],
        "advanced": ["cad fashion design","sustainable fashion","supply chain management","e-commerce","brand management","fashion marketing","production management"],
        "tools":    ["clo3d","adobe illustrator","photoshop","ms office","shopify","sap retail"],
        "resources": [
            {"title":"Parsons School of Design Online","type":"🎓 Course","link":"https://www.coursera.org/parsons"},
            {"title":"Vogue Business","type":"📰 Publication","link":"https://www.voguebusiness.com"},
        ]
    },
    "ARTS": {
        "core":     ["fine arts","illustration","drawing","painting","portfolio development","art history","creative direction","exhibition","community art","art education"],
        "advanced": ["digital art","3d modeling","animation","video production","nft art","gallery management","art therapy","grant writing","public art installations"],
        "tools":    ["adobe photoshop","illustrator","procreate","blender","after effects","canva","unity"],
        "resources": [
            {"title":"Skillshare Art Classes","type":"🏋️ Platform","link":"https://www.skillshare.com"},
            {"title":"Domestika","type":"🏋️ Platform","link":"https://www.domestika.org"},
        ]
    },
    "PUBLIC-RELATIONS": {
        "core":     ["press releases","media relations","crisis communication","copywriting","social media","event management","brand management","stakeholder communication","journalism","seo"],
        "advanced": ["digital pr","influencer marketing","data-driven pr","podcast pr","employee advocacy","reputation management","thought leadership","analytics","content strategy"],
        "tools":    ["cision","meltwater","sprout social","canva","ms office","google analytics","mailchimp"],
        "resources": [
            {"title":"PRSA Accreditation (APR)","type":"📜 Cert","link":"https://www.prsa.org/professional-development/apr"},
            {"title":"Coursera PR Courses","type":"🎓 Course","link":"https://www.coursera.org/courses?query=public+relations"},
        ]
    },
    "BPO": {
        "core":     ["customer service","call center","crm","data entry","communication","problem solving","technical support","quality assurance","kpi tracking","escalation management"],
        "advanced": ["workforce management","six sigma","rpa","chatbot implementation","omnichannel support","bpo operations","client management","sla management"],
        "tools":    ["salesforce","zendesk","freshdesk","avaya","genesys","excel","ms teams"],
        "resources": [
            {"title":"HDI Customer Service Rep","type":"📜 Cert","link":"https://www.thinkhdi.com/certification/customer-service-representative.aspx"},
            {"title":"Coursera Customer Service","type":"🎓 Course","link":"https://www.coursera.org/courses?query=customer+service"},
        ]
    },
}

# ══════════════════════════════════════════════════════════
#  EXACT CATEGORY MAP — must match Kaggle df['Category'] values
# ══════════════════════════════════════════════════════════
CATEGORY_MAP = {cat: cat for cat in CAREER_SKILLS}
# Also handle any display-name variants returned by the model
CATEGORY_MAP.update({
    "INFORMATION-TECHNOLOGY": "INFORMATION-TECHNOLOGY",
    "ENGINEERING":             "ENGINEERING",
    "DESIGNER":                "DESIGNER",
    "DIGITAL-MEDIA":           "DIGITAL-MEDIA",
    "CONSULTANT":              "CONSULTANT",
    "ACCOUNTANT":              "ACCOUNTANT",
    "FINANCE":                 "FINANCE",
    "BANKING":                 "BANKING",
    "HR":                      "HR",
    "BUSINESS-DEVELOPMENT":    "BUSINESS-DEVELOPMENT",
    "SALES":                   "SALES",
    "HEALTHCARE":              "HEALTHCARE",
    "FITNESS":                 "FITNESS",
    "TEACHER":                 "TEACHER",
    "ADVOCATE":                "ADVOCATE",
    "CHEF":                    "CHEF",
    "CONSTRUCTION":            "CONSTRUCTION",
    "AUTOMOBILE":              "AUTOMOBILE",
    "AVIATION":                "AVIATION",
    "AGRICULTURE":             "AGRICULTURE",
    "APPAREL":                 "APPAREL",
    "ARTS":                    "ARTS",
    "PUBLIC-RELATIONS":        "PUBLIC-RELATIONS",
    "BPO":                     "BPO",
})

# Friendly display names for UI
FRIENDLY_NAMES = {
    "INFORMATION-TECHNOLOGY": "Information Technology",
    "ENGINEERING":             "Engineering",
    "DESIGNER":                "UI/UX Designer",
    "DIGITAL-MEDIA":           "Digital Media & Marketing",
    "CONSULTANT":              "Consultant",
    "ACCOUNTANT":              "Accountant",
    "FINANCE":                 "Finance",
    "BANKING":                 "Banking",
    "HR":                      "Human Resources",
    "BUSINESS-DEVELOPMENT":    "Business Development",
    "SALES":                   "Sales",
    "HEALTHCARE":              "Healthcare",
    "FITNESS":                 "Fitness & Wellness",
    "TEACHER":                 "Teacher / Educator",
    "ADVOCATE":                "Legal / Advocate",
    "CHEF":                    "Chef / Culinary Arts",
    "CONSTRUCTION":            "Construction",
    "AUTOMOBILE":              "Automobile",
    "AVIATION":                "Aviation",
    "AGRICULTURE":             "Agriculture",
    "APPAREL":                 "Fashion / Apparel",
    "ARTS":                    "Arts & Creative",
    "PUBLIC-RELATIONS":        "Public Relations",
    "BPO":                     "BPO / Customer Service",
}

def friendly(cat): return FRIENDLY_NAMES.get(cat, cat.replace("-"," ").title())

# ══════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════
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
    csv_file = next((os.path.join(r,f) for r,_,fs in os.walk(path) for f in fs if f.endswith(".csv")), None)
    df = pd.read_csv(csv_file)
    df.dropna(subset=['Resume_str','Category'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stops = set(stopwords.words('english'))
    lem   = WordNetLemmatizer()

    def clean(t):
        t = re.sub(r'http\S+|www\S+|<.*?>|[^a-zA-Z\s]', ' ', str(t)).lower()
        return ' '.join([lem.lemmatize(w) for w in t.split() if w not in stops and len(w) > 2])

    df['clean'] = df['Resume_str'].apply(clean)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Category'])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X = tfidf.fit_transform(df['clean'])
    lr = LogisticRegression(max_iter=1000, C=5, random_state=42)
    lr.fit(X, df['label'])

    bert = SentenceTransformer('all-MiniLM-L6-v2')
    def trunc(t, n=150): return ' '.join(str(t).split()[:n])

    profiles = {}
    for cat in df['Category'].unique():
        docs = df[df['Category']==cat]['clean'].apply(trunc).tolist()
        emb  = bert.encode(docs, batch_size=32, show_progress_bar=False)
        profiles[cat] = emb.mean(axis=0)

    names = list(profiles.keys())
    vecs  = np.array(list(profiles.values()))
    return {"clean": clean, "tfidf": tfidf, "lr": lr, "le": le,
            "bert": bert, "names": names, "vecs": vecs, "trunc": trunc}

# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def pdf_text(b):
    try:
        import pdfplumber
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
            t.write(b); tp = t.name
        txt = ""
        with pdfplumber.open(tp) as pdf:
            for pg in pdf.pages: txt += (pg.extract_text() or "") + "\n"
        os.unlink(tp)
        return txt.strip()
    except ImportError:
        st.warning("Run: pip install pdfplumber"); return ""
    except Exception as e:
        st.error(f"PDF error: {e}"); return ""

def extract_skills(resume):
    rl = resume.lower(); found = set()
    for cd in CAREER_SKILLS.values():
        for k, lst in cd.items():
            if isinstance(lst, list) and lst and isinstance(lst[0], str):
                for s in lst:
                    if s.lower() in rl: found.add(s)
    return found

def skill_gap(user_skills, career_key):
    key = CATEGORY_MAP.get(career_key, career_key)
    if key not in CAREER_SKILLS:
        return {}, []
    cd = CAREER_SKILLS[key]
    ul = {s.lower() for s in user_skills}
    core  = set(cd.get("core", []))
    adv   = set(cd.get("advanced", []))
    tools = set(cd.get("tools", []))
    return {
        "have_core":  sorted(core & ul),
        "miss_core":  sorted(core - ul),
        "have_adv":   sorted(adv & ul),
        "miss_adv":   sorted(adv - ul),
        "have_tools": sorted(tools & ul),
        "miss_tools": sorted(tools - ul),
        "match_pct":  round(len(core & ul) / max(len(core), 1) * 100),
    }, cd.get("resources", [])

def bert_recs(m, text, n=5):
    c = m["clean"](text)
    v = m["bert"].encode([m["trunc"](c)])
    s = cosine_similarity(v, m["vecs"])[0]
    idx = s.argsort()[::-1][:n]
    return [{"career": m["names"][i], "score": round(float(s[i])*100, 1)} for i in idx]

def tfidf_pred(m, text):
    c = m["clean"](text)
    v = m["tfidf"].transform([c])
    return m["le"].inverse_transform(m["lr"].predict(v))[0]

def jd_match(resume, jd):
    rw = set(re.findall(r'\b[a-z][a-z0-9+#.]{1,}\b', resume.lower()))
    jw = set(re.findall(r'\b[a-z][a-z0-9+#.]{1,}\b', jd.lower()))
    stop = {"and","the","for","with","you","are","our","have","will","to","of","in","a","an",
            "is","be","as","we","or","on","at","by","this","that","your","their","more","from"}
    jw -= stop
    if not jw: return 0, set(), set()
    matched = rw & jw
    return round(len(matched)/len(jw)*100, 1), matched, jw-rw

# ══════════════════════════════════════════════════════════
#  DEMO RESUMES — crafted to match actual Kaggle categories
# ══════════════════════════════════════════════════════════
DEMOS = {
    "🖥️ Information Technology (ML/AI)": """
Cherry Agarwal | ML & IT Engineer | cherry@email.com | Nagpur, India

SUMMARY
Machine learning and information technology engineer with hands-on experience building
NLP, computer vision, and deep learning systems. Strong foundation in Python, TensorFlow,
PyTorch, scikit-learn, transformers, and cloud infrastructure.

SKILLS
Languages: Python, SQL, R, Java
ML/AI: TensorFlow, PyTorch, scikit-learn, BERT, transformers, NLP, computer vision,
       deep learning, neural networks, machine learning, feature engineering, statistics
Data: pandas, numpy, matplotlib, seaborn, spark, a/b testing, data visualization
Cloud/MLOps: Docker, Kubernetes, AWS, linux, networking, git, MLflow, Hugging Face
Tools: Jupyter, vscode, github, postman, jira, azure, gcp

EXPERIENCE
ML Engineer Intern — TechCorp AI (2023)
• Built NLP classification using BERT + transformers — 94% accuracy
• Deployed models using Docker + Kubernetes on AWS reducing latency 40%
• Data pipelines with pandas, numpy processing 1M+ daily records
• Computer vision defect detection model in PyTorch, TensorFlow

EDUCATION
B.Tech Computer Science — VNIT Nagpur (2020–2024) | CGPA: 8.7
CERTIFICATIONS: Google TensorFlow Developer, AWS ML Specialty
""",
    "💼 Business Development": """
Rahul Mehta | Business Development Manager | rahul@email.com

SUMMARY
Results-driven business development professional with 5 years experience in
sales, lead generation, CRM management, and partnership development.

SKILLS
Core: sales, business development, crm, negotiation, lead generation, market research,
      excel, powerpoint, networking, cold calling, communication, customer relationship
Advanced: salesforce, hubspot, account management, financial modeling, sql, data analysis,
          product strategy, partnership development

EXPERIENCE
Business Development Manager — StartupXYZ (2020–2024)
• Generated $2M pipeline through strategic lead generation and cold calling
• Managed Salesforce CRM for team of 10, improving conversion by 25%
• Built partnership network of 50+ companies through negotiation

EDUCATION
MBA Marketing — IIM Nagpur (2019) | B.Com — Mumbai University (2017)
""",
    "🎨 UI/UX Designer": """
Priya Sharma | UI/UX Designer | priya@email.com

SUMMARY
Creative UI/UX designer with 4 years building beautiful digital products.
Expert in Figma, user research, and design systems.

SKILLS
Core: figma, adobe xd, photoshop, illustrator, ui design, ux design, typography,
      color theory, responsive design, wireframing, prototyping, user research
Advanced: motion design, after effects, design systems, accessibility, animation,
          3d modeling, blender

EXPERIENCE
UI/UX Designer — DesignStudio (2021–2024)
• Designed 20+ mobile apps and web platforms in Figma
• Led user research and usability testing for fintech product
• Built design system used by 15 engineers

EDUCATION
B.Des Visual Communication — NID Ahmedabad (2021)
""",
    "🏥 Healthcare Professional": """
Dr. Anita Singh | Healthcare Professional | anita@email.com

SUMMARY
Dedicated healthcare professional with expertise in patient care,
clinical documentation, and medical terminology.

SKILLS
Core: patient care, medical terminology, emr, hipaa, clinical documentation,
      nursing, diagnosis, treatment planning, medication management, vital signs
Advanced: telemedicine, healthcare analytics, medical coding, icd-10, care coordination

TOOLS: epic, cerner, ms office

EXPERIENCE
Clinical Coordinator — City Hospital (2020–2024)
• Managed patient care for 30+ daily patients
• EMR documentation in Epic with 100% compliance
• Coordinated telemedicine consultations during COVID-19

EDUCATION
MBBS — AIIMS Delhi (2019) | MD Internal Medicine — (2022)
""",
}

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sb-logo'>🔭 Career<span>Lens</span></div><div class='sb-ver'>v3.0 · Fixed + Enhanced</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='sb-sec'>Input Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["📄 Upload PDF/TXT", "✏️ Paste Resume", "🎮 Demo Profile"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    top_n = st.slider("Career matches to show", 3, 10, 5)
    st.markdown("<hr>", unsafe_allow_html=True)
    enable_jd = st.checkbox("🔗 JD Matcher", value=False, help="Match resume keywords against a job description")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.73rem;color:rgba(148,163,184,.48);line-height:1.78'>
      <div style='color:rgba(165,180,252,.62);font-weight:600;margin-bottom:.35rem;font-size:.76rem'>How it works</div>
      ① Upload or paste your resume<br>
      ② BERT encodes semantic meaning<br>
      ③ TF-IDF + LR classifies keywords<br>
      ④ Skill gap + roadmap generated
      <br><br>
      <span style='color:rgba(99,102,241,.42);font-size:.62rem;font-family:"JetBrains Mono",monospace'>
        Model: all-MiniLM-L6-v2<br>
        Dataset: Kaggle — 24 categories<br>
        2484 resumes · 5000 TF-IDF features
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-stats'>
    <div><div class='hs-num'>24</div><div class='hs-lbl'>Categories</div></div>
    <div><div class='hs-num'>BERT</div><div class='hs-lbl'>Semantic AI</div></div>
    <div><div class='hs-num'>NLP</div><div class='hs-lbl'>Powered</div></div>
  </div>
  <div class='hero-eye'>AI-Powered Career Intelligence — v3.0 Fixed</div>
  <div class='hero-title'>Decode Your<br><span>Career Path</span></div>
  <div class='hero-sub'>
    Upload your resume. Get instant career matches, skill gap analysis,
    and a personalized learning roadmap powered by BERT + TF-IDF NLP.<br>
    <span style='font-size:.85rem;color:rgba(52,211,153,.7)'>
    ✅ Now correctly maps all 24 Kaggle resume categories
    </span>
  </div>
  <div class='hero-pills'>
    <span class='hero-pill'>🤖 BERT Embeddings</span>
    <span class='hero-pill'>📊 TF-IDF + Logistic Regression</span>
    <span class='hero-pill'>🎯 Skill Gap (All 24 categories)</span>
    <span class='hero-pill'>🗺️ Learning Roadmap</span>
    <span class='hero-pill'>🔗 JD Matcher</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  INPUT
# ══════════════════════════════════════════════════════════
resume_text = ""

if mode == "📄 Upload PDF/TXT":
    st.markdown("<div class='sec-eye'>Step 01</div><div class='sec-head'>Upload your resume</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop PDF or TXT", type=["pdf","txt"], label_visibility="collapsed")
    if uploaded:
        if uploaded.type == "application/pdf":
            resume_text = pdf_text(uploaded.read())
            if not resume_text:
                uploaded.seek(0); resume_text = uploaded.read().decode("utf-8", errors="ignore")
        else:
            resume_text = uploaded.read().decode("utf-8", errors="ignore")
        if resume_text:
            wc = len(resume_text.split())
            st.markdown(f"<div class='wc-badge'>✅ {wc} words extracted</div>", unsafe_allow_html=True)
            with st.expander("👁 Preview extracted text"):
                st.code(resume_text[:2000] + ("…" if len(resume_text)>2000 else ""), language=None)
        else:
            st.error("Could not extract text. Try Paste Text option.")

elif mode == "✏️ Paste Resume":
    st.markdown("<div class='sec-eye'>Step 01</div><div class='sec-head'>Paste your resume</div>", unsafe_allow_html=True)
    resume_text = st.text_area("Resume", placeholder="Paste your full resume — skills, experience, projects, education…", height=320, label_visibility="collapsed")

else:
    st.markdown("<div class='sec-eye'>Step 01</div><div class='sec-head'>Choose demo profile</div>", unsafe_allow_html=True)
    sel = st.selectbox("Demo", list(DEMOS.keys()), label_visibility="collapsed")
    resume_text = DEMOS[sel]
    st.info("✅ Demo loaded — click Analyze below")
    with st.expander("👁 View demo resume"):
        st.code(resume_text, language=None)

jd_text = ""
if enable_jd:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='sec-eye'>Optional — JD Match</div><div class='sec-head'>Paste Job Description</div>", unsafe_allow_html=True)
    jd_text = st.text_area("JD", placeholder="Paste the job description you're targeting…", height=180, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)
cb, ch = st.columns([2,5])
with cb:
    analyze = st.button("🔭 Analyze My Resume", use_container_width=True)
with ch:
    st.markdown("""<div style='padding:.65rem 0;font-size:.76rem;color:rgba(148,163,184,.38);
    font-family:"JetBrains Mono",monospace'>First run ~60-90s (downloads BERT ~90MB + Kaggle dataset). Instant after.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════
if analyze and resume_text.strip():
    with st.spinner("⚙️ Loading AI models (first run ~60-90s)…"):
        try:
            m = load_models()
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.code("pip install nltk sentence-transformers kagglehub scikit-learn pdfplumber pandas numpy torch transformers")
            st.stop()

    with st.spinner("🔬 Analyzing resume…"):
        tp     = tfidf_pred(m, resume_text)
        recs   = bert_recs(m, resume_text, n=top_n)
        skills = extract_skills(resume_text)
        top_c  = recs[0]["career"] if recs else tp
        gd, res = skill_gap(skills, top_c)

    st.markdown("<hr>", unsafe_allow_html=True)

    match_pct  = gd.get("match_pct", 0) if isinstance(gd, dict) else 0
    miss_count = len(gd.get("miss_core", [])) if isinstance(gd, dict) else 0

    # ── METRICS ─────────────────────────────────────────
    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric-tile'><div class='mt-icon'>🎯</div><div class='mt-num'>{len(skills)}</div><div class='mt-lbl'>Skills Detected</div></div>
      <div class='metric-tile'><div class='mt-icon'>📊</div><div class='mt-num'>{match_pct}%</div><div class='mt-lbl'>Core Skill Match</div></div>
      <div class='metric-tile'><div class='mt-icon'>📚</div><div class='mt-num'>{miss_count}</div><div class='mt-lbl'>Skills to Learn</div></div>
      <div class='metric-tile'><div class='mt-icon'>🔗</div><div class='mt-num'>{len(res)}</div><div class='mt-lbl'>Resources Found</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── JD MATCH ─────────────────────────────────────────
    if enable_jd and jd_text.strip():
        js, jmatch, jmiss = jd_match(resume_text, jd_text)
        top_miss = sorted(list(jmiss), key=len, reverse=True)[:18]
        miss_html = "".join([f"<span class='pm'>{w}</span>" for w in top_miss])
        st.markdown(f"""
        <div class='jd-box'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem'>
            <div>
              <div class='sec-eye'>JD Match Analysis</div>
              <div style='font-family:"Syne",sans-serif;font-size:1.1rem;font-weight:700;color:#f1f5f9'>Resume ↔ Job Description</div>
            </div>
            <div style='font-family:"Syne",sans-serif;font-size:2.2rem;font-weight:800;
              background:linear-gradient(135deg,#818cf8,#34d399);-webkit-background-clip:text;
              -webkit-text-fill-color:transparent;background-clip:text'>{js}%</div>
          </div>
          <div class='jd-track'><div class='jd-fill' style='width:{min(js,100)}%'></div></div>
          <div style='font-size:.72rem;color:rgba(148,163,184,.4);font-family:"JetBrains Mono",monospace;margin-top:.35rem'>
            {len(jmatch)} keywords matched · {len(jmiss)} keywords missing
          </div>
          {"<div style='margin-top:1rem'><div style='font-size:.68rem;font-weight:700;color:rgba(248,113,113,.7);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem'>Add these to your resume</div>" + miss_html + "</div>" if top_miss else ""}
        </div>
        """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🎯 Career Matches", "🔍 Skill Gap", "📚 Roadmap", "📊 Deep Analysis"])

    def pills(items, cls):
        return "".join([f"<span class='{cls}'>{s}</span>" for s in items])

    # ── TAB 1: CAREER MATCHES ─────────────────────────────
    with t1:
        cl, cr = st.columns([1.2,1])
        with cl:
            st.markdown("<div class='sec-eye'>BERT Semantic Similarity</div><div class='sec-head'>Your career matches</div>", unsafe_allow_html=True)
            for i, r in enumerate(recs):
                top_cls = "top" if i==0 else ""
                rk_cls  = "gld" if i==0 else ""
                rk_lbl  = "⚡ Best Match" if i==0 else f"#{i+1}"
                st.markdown(f"""
                <div class='mc {top_cls}'>
                  <div class='mc-rank {rk_cls}'>{rk_lbl}</div>
                  <div class='mc-name'>{friendly(r["career"])}</div>
                  <div class='mc-score'>{r["score"]}% semantic similarity · <span style='color:rgba(99,102,241,.7);font-family:"JetBrains Mono",monospace;font-size:.7rem'>{r["career"]}</span></div>
                  <div class='bar-bg'><div class='bar-fg' style='width:{min(r["score"],100)}%'></div></div>
                </div>""", unsafe_allow_html=True)
        with cr:
            st.markdown("<div class='sec-eye'>TF-IDF Classifier</div><div class='sec-head'>ML prediction</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='pred-box'>
              <div class='pred-lbl'>Logistic Regression · TF-IDF</div>
              <div class='pred-val'>{friendly(tp)}</div>
              <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;color:rgba(99,102,241,.6);margin-top:.2rem'>{tp}</div>
              <div style='font-size:.74rem;color:rgba(148,163,184,.42);margin-top:.3rem'>Keyword-based classification</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='sec-eye'>Detected Skills</div>", unsafe_allow_html=True)
            if skills:
                st.markdown(f"<div style='line-height:2.3'>{pills(sorted(skills)[:24],'ph')}</div>", unsafe_allow_html=True)
                if len(skills) > 24: st.caption(f"+{len(skills)-24} more detected")
            else:
                st.markdown("""<div class='no-data-box'>
                ⚠️ No skills detected. Add more specific keywords to your resume
                (e.g. Python, Excel, SQL, Figma, etc.)
                </div>""", unsafe_allow_html=True)

    # ── TAB 2: SKILL GAP ──────────────────────────────────
    with t2:
        st.markdown("<div class='sec-eye'>Skill Gap Analysis</div><div class='sec-head'>What you have vs. what you need</div>", unsafe_allow_html=True)

        career_opts = [r["career"] for r in recs]
        display_opts = [f"{friendly(c)} ({c})" for c in career_opts]
        sel_idx = st.selectbox("Analyze gap for:", range(len(career_opts)),
                                format_func=lambda i: display_opts[i], key="gap_sel")
        sel_c = career_opts[sel_idx]
        gd2, res2 = skill_gap(skills, sel_c)

        if isinstance(gd2, dict) and gd2:
            m2 = gd2.get("match_pct", 0)
            have_total = len(gd2["have_core"]) + len(gd2["have_adv"]) + len(gd2["have_tools"])
            miss_total = len(gd2["miss_core"]) + len(gd2["miss_adv"]) + len(gd2["miss_tools"])

            st.markdown(f"""
            <div class='gap-box'>
              <div class='gap-row'>
                <span>Core skill match — <strong>{friendly(sel_c)}</strong></span>
                <span class='gap-pct'>{m2}%</span>
              </div>
              <div class='gap-track'><div class='gap-fill' style='width:{m2}%'></div></div>
              <div style='font-size:.7rem;color:rgba(148,163,184,.38);font-family:"JetBrains Mono",monospace;margin-top:.3rem'>
                {len(gd2["have_core"])} / {len(gd2["have_core"])+len(gd2["miss_core"])} core skills &nbsp;·&nbsp;
                {have_total} skills matched &nbsp;·&nbsp; {miss_total} skills to acquire
              </div>
            </div>
            """, unsafe_allow_html=True)

            lbl = lambda txt, clr: f"<div style='font-size:.6rem;font-weight:700;color:{clr};letter-spacing:.1em;text-transform:uppercase;margin:.55rem 0 .3rem;font-family:\"JetBrains Mono\",monospace'>{txt}</div>"
            cc1, cc2, cc3 = st.columns(3)

            with cc1:
                st.markdown("<div class='skill-col-head'>🔵 Core Skills</div>", unsafe_allow_html=True)
                if gd2["have_core"]:
                    st.markdown(lbl("✅ YOU HAVE", "rgba(52,211,153,.7)") + f"<div style='line-height:2.2'>{pills(gd2['have_core'],'ph')}</div>", unsafe_allow_html=True)
                if gd2["miss_core"]:
                    st.markdown(lbl("❌ MISSING", "rgba(248,113,113,.7)") + f"<div style='line-height:2.2'>{pills(gd2['miss_core'],'pm')}</div>", unsafe_allow_html=True)
                if not gd2["have_core"] and not gd2["miss_core"]:
                    st.caption("No core skill data")

            with cc2:
                st.markdown("<div class='skill-col-head'>🟣 Advanced Skills</div>", unsafe_allow_html=True)
                if gd2["have_adv"]:
                    st.markdown(lbl("✅ YOU HAVE", "rgba(52,211,153,.7)") + f"<div style='line-height:2.2'>{pills(gd2['have_adv'],'ph')}</div>", unsafe_allow_html=True)
                if gd2["miss_adv"]:
                    st.markdown(lbl("📖 TO LEARN", "rgba(251,191,36,.7)") + f"<div style='line-height:2.2'>{pills(gd2['miss_adv'],'pl')}</div>", unsafe_allow_html=True)

            with cc3:
                st.markdown("<div class='skill-col-head'>🟠 Tools & Platforms</div>", unsafe_allow_html=True)
                if gd2["have_tools"]:
                    st.markdown(lbl("✅ YOU KNOW", "rgba(52,211,153,.7)") + f"<div style='line-height:2.2'>{pills(gd2['have_tools'],'ph')}</div>", unsafe_allow_html=True)
                if gd2["miss_tools"]:
                    st.markdown(lbl("🛠️ TO ADD", "rgba(251,191,36,.7)") + f"<div style='line-height:2.2'>{pills(gd2['miss_tools'],'pl')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='no-data-box'>
            ⚠️ The career <strong>{friendly(sel_c)}</strong> ({sel_c}) matched from your resume
            but isn't in the skill database yet. This can happen when the model matches you to
            a less common category. Try selecting a different career from the dropdown above.
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: ROADMAP ────────────────────────────────────
    with t3:
        st.markdown(f"<div class='sec-eye'>Personalized Learning Path</div><div class='sec-head'>Road to {friendly(top_c)}</div>", unsafe_allow_html=True)
        gf, rf = skill_gap(skills, top_c)

        if isinstance(gf, dict) and gf:
            prio = gf.get("miss_core", [])[:5]
            adv  = gf.get("miss_adv",  [])[:5]

            def step(n, title, sub):
                return f"<div class='step-row'><div class='step-n'>{n}</div><div><div class='step-t'>{title}</div><div class='step-s'>{sub}</div></div></div>"

            # Phase 1
            st.markdown(step("1", "Fill core skill gaps", f"Must-haves for {friendly(top_c)}"), unsafe_allow_html=True)
            if prio:
                st.markdown(f"<div style='margin-left:2.8rem;line-height:2.3;margin-bottom:1rem'>{pills(prio,'pm')}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 You already have all core skills for this role!")
            st.markdown("<br>", unsafe_allow_html=True)

            # Phase 2
            st.markdown(step("2", "Level up with advanced skills", "Differentiate yourself from other candidates"), unsafe_allow_html=True)
            if adv:
                st.markdown(f"<div style='margin-left:2.8rem;line-height:2.3;margin-bottom:1rem'>{pills(adv,'pl')}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 You're already advanced in this field!")
            st.markdown("<br>", unsafe_allow_html=True)

            # Phase 3
            st.markdown(step("3", f"Curated learning resources", f"Handpicked for {friendly(top_c)}"), unsafe_allow_html=True)
            if rf:
                st.markdown("<div style='margin-left:2.8rem;margin-bottom:1.2rem'>", unsafe_allow_html=True)
                for r in rf:
                    st.markdown(f"""
                    <div class='res-card'>
                      <div class='res-t'>{r["title"]}</div>
                      <div class='res-m'>{r["type"]} &nbsp;·&nbsp; <a href='{r["link"]}' target='_blank' style='color:#818cf8;text-decoration:none'>Open resource ↗</a></div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("No specific resources listed for this category yet.")
            st.markdown("<br>", unsafe_allow_html=True)

            # Phase 4
            st.markdown(step("4", "Get hired — action plan", "Apply your new skills and get noticed"), unsafe_allow_html=True)
            for icon, txt in [
                ("🔨", "Build 2-3 portfolio projects and push to GitHub"),
                ("🐙", "Contribute to open source repositories in your domain"),
                ("📝", "Update LinkedIn with your new skills and project links"),
                ("🤝", "Join online communities — Discord, Reddit, Slack groups"),
                ("💼", "Apply with a tailored resume for each job (use JD Matcher!)"),
                ("🏅", "Get 1-2 certifications to add credibility"),
            ]:
                st.markdown(f"<div class='ns-item'><span>{icon}</span><span>{txt}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='no-data-box'>
            ⚠️ <strong>{friendly(top_c)}</strong> matched from your resume but isn't in the
            knowledge base yet. Try switching to Demo Profile mode and use a more skill-rich resume
            to get a tech/business career match with full roadmap support.
            </div>""", unsafe_allow_html=True)

    # ── TAB 4: DEEP ANALYSIS ─────────────────────────────
    with t4:
        st.markdown("<div class='sec-eye'>Deep Analysis</div><div class='sec-head'>Full Report</div>", unsafe_allow_html=True)

        # BERT scores table
        st.markdown("<div style='font-size:.75rem;font-weight:600;color:rgba(165,180,252,.7);margin-bottom:.5rem;letter-spacing:.06em;text-transform:uppercase'>BERT Similarity Scores</div>", unsafe_allow_html=True)
        bert_df = pd.DataFrame([{
            "Category (Raw)": r["career"],
            "Career (Friendly)": friendly(r["career"]),
            "Semantic Match (%)": r["score"]
        } for r in recs])
        st.dataframe(bert_df, use_container_width=True, hide_index=True)

        # Detected skills
        if skills:
            st.markdown("<br><div style='font-size:.75rem;font-weight:600;color:rgba(165,180,252,.7);margin-bottom:.5rem;letter-spacing:.06em;text-transform:uppercase'>DETECTED SKILLS</div>", unsafe_allow_html=True)
            sk = sorted(skills); n = 3; chunk = len(sk)//n+1
            scols = st.columns(n)
            for i, col in enumerate(scols):
                with col:
                    sub = sk[i*chunk:(i+1)*chunk]
                    if sub: st.dataframe(pd.DataFrame({"Skill": sub}), use_container_width=True, hide_index=True)

        # Multi-career comparison
        st.markdown("<br><div style='font-size:.75rem;font-weight:600;color:rgba(165,180,252,.7);margin-bottom:.5rem;letter-spacing:.06em;text-transform:uppercase'>MULTI-CAREER SKILL COMPARISON</div>", unsafe_allow_html=True)
        table = []
        for r in recs[:5]:
            g, _ = skill_gap(skills, r["career"])
            in_kb = r["career"] in CAREER_SKILLS
            if isinstance(g, dict) and g:
                table.append({
                    "Career": friendly(r["career"]),
                    "Semantic (%)": r["score"],
                    "Core Match (%)": g.get("match_pct", 0),
                    "Skills ✅": len(g.get("have_core",[])) + len(g.get("have_adv",[])),
                    "Skills 📚": len(g.get("miss_core",[])) + len(g.get("miss_adv",[])),
                    "In KB": "✅" if in_kb else "⚠️ Limited"
                })
            else:
                table.append({
                    "Career": friendly(r["career"]),
                    "Semantic (%)": r["score"],
                    "Core Match (%)": 0,
                    "Skills ✅": 0,
                    "Skills 📚": 0,
                    "In KB": "⚠️ Limited"
                })
        if table:
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # Download report
        lines = [
            "CAREERLENS AI — ANALYSIS REPORT (v3.0)", "="*50,
            f"Top Match: {friendly(top_c)} ({top_c})", "="*50,
            f"TF-IDF Prediction: {friendly(tp)} ({tp})",
            f"Skills Detected: {len(skills)}",
            f"Core Skill Match: {match_pct}%", "",
            "BERT CAREER MATCHES:"
        ] + [f"  {friendly(r['career'])} ({r['career']}): {r['score']}%" for r in recs] \
          + ["", "DETECTED SKILLS:"] + [f"  {s}" for s in sorted(skills)] \
          + ["", "CORE SKILL GAPS:"] + [f"  ❌ {s}" for s in gd.get("miss_core", [])]

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇️ Download Full Report", "\n".join(lines), "careerlens_report.txt", "text/plain")

elif analyze and not resume_text.strip():
    st.warning("⚠️ Please provide your resume first.")

else:
    st.markdown("""
    <div style='text-align:center;padding:5rem 1rem 3rem'>
      <div style='font-size:4rem;margin-bottom:1.5rem'>🔭</div>
      <div style='font-family:"Syne",sans-serif;font-size:1.6rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;margin-bottom:.6rem'>Ready to analyze</div>
      <div style='font-size:.88rem;color:rgba(148,163,184,.48);max-width:360px;margin:0 auto;line-height:1.7'>
        Upload your resume or pick a demo,<br>
        then click <span style='color:#818cf8;font-weight:600'>Analyze My Resume</span>
      </div>
    </div>
    <div class='land-grid'>
      <div class='land-f'><div class='lf-icon'>🤖</div><div class='lf-t'>BERT Semantic Matching</div>
        <div class='lf-d'>Deep NLP embeddings compare your resume to all 24 career profiles using cosine similarity</div></div>
      <div class='land-f'><div class='lf-icon'>📊</div><div class='lf-t'>TF-IDF Classification</div>
        <div class='lf-d'>Logistic regression on keyword TF-IDF vectors gives an instant direct prediction</div></div>
      <div class='land-f'><div class='lf-icon'>🎯</div><div class='lf-t'>Skill Gap Analysis</div>
        <div class='lf-d'>Exact skills you have vs. what each of the 24 careers requires — core, advanced, tools</div></div>
      <div class='land-f'><div class='lf-icon'>🗺️</div><div class='lf-t'>Personalized Roadmap</div>
        <div class='lf-d'>4-phase learning plan with curated books, courses, certifications and action steps</div></div>
    </div>
    """, unsafe_allow_html=True)
