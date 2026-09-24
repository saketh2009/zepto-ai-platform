# Zepto Data & AI Platform Capstone

An end-to-end multi-tier platform combining data engineering, predictive analytics, and a grounded GenAI support assistant.

## Repository Layout
- `/data_pipeline`: Scraping pipeline, relational SQLite schema, and query validation.
- `/analytics`: EDA, pipeline-based ML training, evaluations, and linear regression.
- `/support_assistant`: ChromaDB document indexing, LangGraph intent router, and FastAPI service.
- `requirements.txt`: Single unified requirements manifest.

---

## Module 1: Data Pipeline
- **Scraping**: Fetches 100 books from `books.toscrape.com` using requests and BeautifulSoup.
- **Currency Conversion**: Fixed baseline constant of 1 GBP = 105.50 INR.
- **Relational Store**: Normalized SQLite database `zepto_catalog.db` with PK/FK table relationships (`categories` and `books`).
- **Validation**: Equivalence verified between SQL JOIN and in-memory `pd.merge`.

---

## Module 2: Analytics Pipeline
- **Profiling & Missing Data**:
  - `embarked` (<5% missing): Imputed with mode.
  - `age` (~19.8% missing): Imputed with median.
  - `deck` (>70% missing): Dropped due to low reliability.
- **Skewness**: `fare` is positively skewed (mean > median > mode).
- **Outliers**: Identified via IQR boundaries for both age and fare.
- **Predictive ML**: Trained Logistic Regression, Decision Tree, and Random Forest models using strict train-only `ColumnTransformer` preprocessing. Imbalance handled using SMOTE and balanced class weighting.
- **Regression Side-Task**: Linear regression predicting fare with heteroscedasticity verified via residual plot analysis.

---

## Module 3: Support Assistant (RAG Pipeline Architecture)

### 1. Architectural Flow
- **Ingestion**: All 8 policy documents (`support_assistant/docs/doc_01.txt` to `doc_08.txt`) are loaded into memory.
- **Embedding**: Document chunks are embedded locally using `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed inside an in-memory ChromaDB collection (`zepto_policies`).
- **Routing & Retrieval**: A LangGraph `StateGraph` routes requests:
  - `classify_intent`: Checks for keywords (`delivery`, `refund`, `tracking`, etc.) to classify queries into `policy_question` or `general_question`.
  - `retrieve_and_answer`: Queries ChromaDB for top-3 relevant chunks using cosine similarity.
  - `direct_answer`: Handles out-of-scope non-policy queries.
- **Generation & Mock Toggle**:
  - `MOCK_LLM=1` (default): Answers deterministically using `f"Based on the retrieved context: {top_chunk_snippet}"` with no external API calls.
  - `MOCK_LLM=0`: Uses the structured few-shot prompt template and routes to a real LLM API with schema retry logic.

### 2. Recorded FastAPI Example Responses (MOCK_LLM=1)

**Call 1: Policy Question (Triggering Retrieval)**
- **Endpoint**: `POST /ask`
- **Request Payload**:
  ```json
  {"query": "What are your delivery fees and thresholds?"}
- **Response**:
  ```json
  {
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery",
    "sources": ["doc_01"],
    "confidence": 1.0
  }
  **Call 2: General Question (Direct Answer)**
- **Endpoint**: `POST /ask`
- **Request Payload**:
  ```json
  {"query": "What is the capital of France?"}
  - **Response**:
  {
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
### 3. Container Execution
Build and execute the local Docker container via:
```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run -p 7860:7860 zepto-support-assistant
```
  
