import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

try:
    print("Connecting to database")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected successfully!\n")
    
    cursor.execute("""
        SELECT job_id, title, description 
        FROM jobs 
        ORDER BY job_id
    """)
    
    jobs = cursor.fetchall()
    
    print("ALL JOBS IN DATABASE")
    
    if jobs:
        for job in jobs:
            job_id, title, description = job
            print(f"\nJob ID: {job_id}")
            print(f"Title: {title}")
            print(f"Description: {description}")
        print(f"\nTotal jobs: {len(jobs)}")
    else:
        print("No jobs found in database.")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("\nDatabase connection closed.")
