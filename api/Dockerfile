FROM python:3.11-slim
WORKDIR /app
# CPU-only torch first, so sentence-transformers does not pull the large CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt
COPY api/ api/
COPY data_kit/sample_products.json data_kit/sample_products.json
ENV PYTHONPATH=/app/api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
