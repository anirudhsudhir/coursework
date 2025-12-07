-- Lab-11: Vector Search - PostgreSQL
-- Task 1: Create table 'jobs' to store title, description and embeddings

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop table if exists (for clean setup)
DROP TABLE IF EXISTS jobs;

-- Create jobs table with vector column for embeddings
-- Using 384 dimensions for all-MiniLM-L6-v2 model
CREATE TABLE jobs (
    job_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    embedding vector(384)
);

-- Create an index on the embedding column for efficient vector search
-- Using cosine distance operator (<=>)
CREATE INDEX ON jobs USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Alternative: HNSW index (more accurate but slower to build)
-- CREATE INDEX ON jobs USING hnsw (embedding vector_cosine_ops);

-- Verify table creation
SELECT table_name, column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'jobs';
