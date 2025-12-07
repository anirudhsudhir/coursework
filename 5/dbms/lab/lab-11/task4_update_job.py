import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!\n")

job_id = 2
new_description = "Design interactive dashboards with Power BI and SQL"

try:
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected successfully!\n")
    
    cursor.execute("SELECT title, description FROM jobs WHERE job_id = %s", (job_id,))
    original = cursor.fetchone()
    
    if original:
        print(f"Original Job (ID {job_id}):")
        print(f"  Title: {original[0]}")
        print(f"  Description: {original[1]}\n")
        
        print("Generating new embedding...")
        new_embedding = model.encode(new_description)
        
        cursor.execute(
            """
            UPDATE jobs 
            SET description = %s, embedding = %s 
            WHERE job_id = %s
            """,
            (new_description, new_embedding.tolist(), job_id)
        )
        
        conn.commit()
        
        cursor.execute("SELECT title, description FROM jobs WHERE job_id = %s", (job_id,))
        updated = cursor.fetchone()
        
        print(f"Job updated successfully!")
        print(f"\nUpdated Job (ID {job_id}):")
        print(f"  Title: {updated[0]}")
        print(f"  Description: {updated[1]}")
    else:
        print(f"Job with ID {job_id} not found.")
    
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
