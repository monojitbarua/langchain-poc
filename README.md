
- Setup pgvector for vector store 

    - run pgvector db
        ```
        docker run -d --name pgvector -e POSTGRES_PASSWORD=password -e POSTGRES_USER=user -e POSTGRES_DB=vector_db -p 5432:5432 ankane/pgvector
        ```
    - install postgres client to access the db for interaction
        ```
        sudo apt install postgresql-client
        ```
    - login to db
        ```
        psql -h localhost -p 5432 -U user -d vector_db
        ```
    - create vector extension
        ```
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
    - create vector db table (might not require initially as PGVector can take default table)
        ```
        CREATE TABLE langchain_vectors (
            id UUID PRIMARY KEY,
            document TEXT,
            metadata JSONB,
            embedding VECTOR(384) -- use 384 for "all-MiniLM-L6-v2"
        );

        CREATE INDEX ON langchain_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        ```
    
    












