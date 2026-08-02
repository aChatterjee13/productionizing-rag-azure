# Azure Functions custom container for the ingestion service.
#
# BUILD CONTEXT MUST BE THE REPOSITORY ROOT, because `ragcore` is a workspace path
# dependency that lives outside this directory:
#
#   docker build -f services/ingestion/Dockerfile -t rag-ingestion:local .
#
# Deployed as a container to an Azure Functions Premium (Elastic) plan, which is what
# the Durable fan-out and the FastEmbed model download need: a Consumption plan's cold
# start and 10-minute ceiling are not enough for a full corpus scan.
FROM mcr.microsoft.com/azure-functions/python:4-python3.13

# AzureFunctionsJobHost__* mirrors host.json for values worth overriding per image.
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# FastEmbed downloads ONNX weights on first use. Pinning the cache inside the image
# keeps a cold start from re-downloading ~600 MB per worker.
ENV RAG_EMBEDDING_CACHE_DIR=/opt/fastembed \
    HF_HOME=/opt/fastembed

# pdfplumber/pypdf need no system packages, but python-pptx and openpyxl benefit from
# a working locale, and libgomp is required by onnxruntime (FastEmbed).
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# ragcore first: it changes less often than the service, so the layer caches well.
COPY packages/ragcore/pyproject.toml packages/ragcore/pyproject.toml
COPY packages/ragcore/ragcore packages/ragcore/ragcore
COPY README.md README.md
RUN pip install --no-deps ./packages/ragcore

# Then the service's third-party requirements.
COPY services/ingestion/requirements.txt services/ingestion/requirements.txt
RUN pip install -r services/ingestion/requirements.txt

# Finally the function app itself, into the script root the host loads from.
WORKDIR /home/site/wwwroot
COPY services/ingestion/host.json ./host.json
COPY services/ingestion/function_app.py ./function_app.py
COPY services/ingestion/ingestion ./ingestion

RUN mkdir -p /opt/fastembed && chmod -R 0777 /opt/fastembed
