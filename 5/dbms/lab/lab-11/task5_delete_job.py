import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

job_id = 4

try:
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("Connected successfully!\n")
    
    cursor.execute("SELECT title, description FROM jobs WHERE job_id = %s", (job_id,))
    job = cursor.fetchone()
    
    if job:
        print(f"Job to delete (ID {job_id}):")
        print(f"  Title: {job[0]}")
        print(f"  Description: {job[1]}\n")
        
        cursor.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.commit()
        
        print(f"Job with ID {job_id} deleted successfully!")
        
        cursor.execute("SELECT COUNT(*) FROM jobs")
        count = cursor.fetchone()[0]
        print(f"\nRemaining jobs in database: {count}")
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
