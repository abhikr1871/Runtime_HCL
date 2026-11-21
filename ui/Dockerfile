FROM python:3.11-slim
WORKDIR /app
COPY ui/requirements.txt ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt
COPY ui/ ui/
COPY eval/ eval/
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
