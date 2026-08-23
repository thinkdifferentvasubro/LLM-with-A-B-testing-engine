FROM python:3.12.9-slim

WORKDIR /app

COPY docker_requirements.txt .

COPY src/ /app/src/
COPY api/ /app/api

RUN pip install --no-cache-dir -r docker_requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('src/RAG/models/all-MiniLM-L6-v2')"

EXPOSE 8080

CMD ["uvicorn", "api.application:app", "--host", "0.0.0.0", "--port", "8080"]