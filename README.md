# Intelligent Candidate Discovery Engine

An end-to-end AI recruitment processing system designed to match, evaluate, and rank candidate profiles against Job Descriptions using context-aware NLP feature vectors and semantic proximity scoring.

## Technical Architecture
- **Feature Extraction:** TF-IDF Vectorization with n-gram structures capturing complex multi-word skills.
- **Matching Metric:** Angular distance mapping via Cosine Similarity calculations.
- **Decision Matrix:** Dual-weight system combining semantic relevance matrices (80%) and industry seniority scaling (20%).

## How to Run
1. Install dependencies: `pip install pandas scikit-learn openpyxl`
2. Run the pipeline execution: `python app.py`
