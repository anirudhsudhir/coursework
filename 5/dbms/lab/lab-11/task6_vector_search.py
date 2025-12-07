import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

print("Loading SentenceTransformer model")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!\n")

user_query = "Looking for data roles with Python and dashboards"
top_k = 3 

try:
    print("Connecting to database")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected successfully!\n")
    
    print(f"User Query: \"{user_query}\"\n")
    print("Generating query embedding")
    query_embedding = model.encode(user_query)
    
    cursor.execute(
        """
        SELECT job_id, title, description, 
               1 - (embedding <=> %s::vector) as similarity
        FROM jobs
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding.tolist(), query_embedding.tolist(), top_k)
    )
    
    results = cursor.fetchall()
    
    print(f"Top {top_k} matching jobs:\n")
    
    for idx, (job_id, title, description, similarity) in enumerate(results, 1):
        print(f"\nRank {idx}:")
        print(f"  Job ID: {job_id}")
        print(f"  Title: {title}")
        print(f"  Description: {description}")
        print(f"  Similarity Score: {similarity:.4f}")
    
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("\nDatabase connection closed.")
