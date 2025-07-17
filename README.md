# Vector Database Stack

A collection of vector databases for development and testing.

## Databases Included

- **ChromaDB** (Port 8000): Open-source embedding database
- **Qdrant** (Port 6333): High-performance vector search engine
- **Weaviate** (Port 8081): ML-first vector database

## Quick Start

```bash
docker-compose up -d
```

That's it! No configuration files needed.

## Services

- ChromaDB: http://localhost:8000
- Qdrant: http://localhost:6333
- Weaviate: http://localhost:8081

## Usage

The databases are ready to use immediately. Example with Python:

```python
# ChromaDB
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)

# Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)

# Weaviate
import weaviate
client = weaviate.Client("http://localhost:8081")
```

## Monitoring

To monitor these databases, use the AURA monitoring stack:
https://github.com/jenastar/AURA

## Data Persistence

All data is persisted in Docker volumes:
- `chromadb_data`
- `qdrant_data`
- `weaviate_data`

## Stop Services

```bash
docker-compose down

# To also remove data:
docker-compose down -v
```