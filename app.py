# app.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_candidate_ranks(job_description, candidates_df):
    """
    Intelligent Candidate Discovery Scoring Engine
    Uses semantic TF-IDF matrix processing and Cosine Similarity mapping
    """
    print("[*] Initializing AI Matching Pipeline...")
    
    # Extract candidate profiles
    corpus = [job_description] + list(candidates_df['Resume_Text'])
    
    # Compute Vector Space Models
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate similarity metrics between Job Description (index 0) and Resumes
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Combine scores with structural parameters
    candidates_df['Semantic_Score'] = similarity_scores
    
    # Weightage formula: 80% Skill/Resume Match + 20% Experience Alignment
    max_exp = candidates_df['Experience_Years'].max() if candidates_df['Experience_Years'].max() > 0 else 1
    candidates_df['Final_Score'] = (candidates_df['Semantic_Score'] * 0.80) + ((candidates_df['Experience_Years'] / max_exp) * 0.20)
    
    # Rank candidates
    candidates_df = candidates_df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    candidates_df['Rank'] = candidates_df.index + 1
    
    # Assign Recommendation Tiers
    conditions = [
        (candidates_df['Final_Score'] >= 0.75),
        (candidates_df['Final_Score'] >= 0.50) & (candidates_df['Final_Score'] < 0.75),
        (candidates_df['Final_Score'] < 0.50)
    ]
    tiers = ['Tier 1 (Strong Match)', 'Tier 2 (Potential Match)', 'Tier 3 (Low Alignment)']
    candidates_df['Recommendation_Tier'] = np.select(conditions, tiers, default='Tier 3')
    
    return candidates_df[['Rank', 'Candidate_ID', 'Candidate_Name', 'Final_Score', 'Experience_Years', 'Recommendation_Tier']]

# Sample Execution Setup for confirmation
if __name__ == "__main__":
    jd = "Seeking a Python Data Scientist with expertise in Natural Language Processing, NLP, Scikit-Learn, and Pandas."
    
    mock_data = {
        'Candidate_ID': ['CAND_01', 'CAND_02', 'CAND_03'],
        'Candidate_Name': ['Aravind Sharma', 'Priya Patel', 'Amit Mishra'],
        'Resume_Text': [
            'Python Data Scientist skilled in NLP, Machine Learning, Scikit-Learn, and Pandas.',
            'Data Engineer with extensive experience in SQL, ETL pipelines, Spark, and cloud architecture.',
            'Java Developer working on web applications, basic SQL databases, and backend services.'
        ],
        'Experience_Years': [4.5, 5.0, 1.5]
    }
    
    df_candidates = pd.DataFrame(mock_data)
    ranked_output = calculate_candidate_ranks(jd, df_candidates)
    
    # Save target output
    ranked_output.to_excel("ranked_candidates.xlsx", index=False)
    print("[+] Intelligent ranking complete. Exported results to ranked_candidates.xlsx")
