FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    STANZA_RESOURCES_DIR=/root/stanza_resources

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY local/tagger-stanza/server.py ./server.py
RUN uv pip install --system fastapi "uvicorn[standard]" python-multipart stanza && \
    python -c "import stanza; stanza.download('lt', processors='tokenize,pos,lemma', package='alksnis')"

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
