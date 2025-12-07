import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

print("Loading SentenceTransformer model")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!\n")

jobs = [
    ("Data Analyst", "Analyze data and build dashboards using Python and SQL."),
    ("BI Developer", "Develop Tableau dashboards and data reports."),
    ("ML Engineer", "Build machine learning models in Python"),
    ("Sales Analyst", "Work on Excel sales data and metrics")
]

try:
    print(f"Connecting to database")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected successfully!\n")
    
    print("Inserting jobs with embeddings")
    for title, description in jobs:
        embedding = model.encode(description)
        
        cursor.execute(
            "INSERT INTO jobs (title, description, embedding) VALUES (%s, %s, %s)",
            (title, description, embedding.tolist())
        )
        print(f"Inserted: {title}")
    
    conn.commit()
    print("\nAll jobs inserted successfully!")
    
    cursor.execute("SELECT COUNT(*) FROM jobs")
    count = cursor.fetchone()[0]
    print(f"Total jobs in database: {count}")
    
except Exception as e:
    print(f"Error: {e}")
    if conn:
        conn.rollback()
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("\nDatabase connection closed.")
