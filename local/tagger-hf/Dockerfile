FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/model

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY local/tagger-hf/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt
COPY local/tagger-hf/server.py ./server.py

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
