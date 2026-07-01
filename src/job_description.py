"""
job_description.py — JD constants derived from the actual job_description.docx.

Role: Senior AI Engineer — Founding Team @ Redrob AI (Series A)
Experience target: 5–9 years (ideal: 6–8 years in applied ML at product companies)
Location: Pune / Noida (India-preferred, relocators welcome)
"""

EMBEDDED_JD = """
Senior AI Engineer — Founding Team at Redrob AI (Series A)

We need someone who owns the intelligence layer of the product: ranking, retrieval,
and matching systems. Production experience with embeddings-based retrieval, vector
databases, hybrid search, and evaluation frameworks for ranking systems.

Must have: embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI
embeddings), vector database or hybrid search (Pinecone, Weaviate, Qdrant, Milvus,
FAISS, Elasticsearch, OpenSearch), Python, evaluation frameworks for ranking
(NDCG, MRR, MAP, A/B testing, offline-to-online correlation).

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost,
LightGBM), HR-tech or marketplace products, distributed systems, open-source
contributions, MLflow, BentoML, MLOps, Hugging Face Transformers.

Shipped: end-to-end ranking, search, or recommendation systems to real users at
meaningful scale. Product company experience required — not pure research, not
pure services/consulting.

Strong signals: recommendation systems, information retrieval, NLP engineering,
search engineering, applied ML at scale. 6–8 years. Active on platform, short
notice period, willing to relocate to Pune or Noida.
"""

# Hard requirements — these are the "absolutely need" skills from the JD
REQUIRED_SKILLS = [
    # Embeddings / retrieval — the core requirement
    "embeddings",
    "sentence transformers",
    "sentence-transformers",
    "information retrieval",
    "vector search",
    "hybrid search",
    "dense retrieval",
    "semantic search",
    # Vector DBs
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    # Ranking / eval
    "ranking",
    "recommendation systems",
    "learning to rank",
    "ndcg",
    "search",
    # Core tech
    "python",
    "machine learning",
    "nlp",
    "hugging face",
    "hugging face transformers",
]

# Nice-to-have — bonus weight
NICE_TO_HAVE_SKILLS = [
    "fine-tuning llms",
    "fine tuning",
    "lora",
    "qlora",
    "peft",
    "xgboost",
    "lightgbm",
    "mlflow",
    "bentoml",
    "mlops",
    "feature engineering",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "a/b testing",
    "prompt engineering",
    "langchain",
    "openai",
    "weights & biases",
    "kubeflow",
    "airflow",
    "spark",
    "kafka",
    "aws",
    "gcp",
    "docker",
    "kubernetes",
]

# Relevant titles — direct matches score highest
TARGET_TITLES = [
    "recommendation systems engineer",
    "search engineer",
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "nlp engineer",
    "applied ml engineer",
    "applied scientist",
    "research engineer",
    "data scientist",
    "software engineer",
    "backend engineer",
    "data engineer",
]

# JD says 5–9 years, ideal is 6–8
MIN_EXPERIENCE_YEARS = 5
MAX_EXPERIENCE_YEARS = 9
IDEAL_MIN_YEARS = 6
IDEAL_MAX_YEARS = 8

# Location preference (Pune, Noida, Indian cities)
PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
    "gurgaon", "gurugram", "chennai", "india"
]

# Consulting-only companies — explicit JD disqualifier
PURE_CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "coforge", "persistent", "mastech"
]

# Product company signals (positive — not consulting)
PRODUCT_COMPANY_SIGNALS = [
    "ai/ml", "fintech", "e-commerce", "food delivery", "software", "saas",
    "transportation", "edtech", "healthtech", "gaming", "media"
]

# Assessment skill keys that are directly JD-relevant
JD_RELEVANT_ASSESSMENTS = [
    "NLP", "Fine-tuning LLMs", "PEFT", "FAISS", "Pinecone",
    "Recommendation Systems", "Feature Engineering", "MLflow",
    "Prompt Engineering", "Data Science"
]
