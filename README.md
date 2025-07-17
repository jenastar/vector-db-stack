# Vector Database Stack

A collection of popular vector databases for development, testing, and experimentation with embeddings and semantic search.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/jenastar/vector-db-stack.git
cd vector-db-stack

# Start all databases
docker-compose up -d

# Verify services are running
docker ps
```

That's it! No configuration files needed. All databases will be available immediately.

## 📦 Included Databases

### ChromaDB (Port 8000)
Open-source embedding database optimized for AI applications.
- Simple API for storing and querying embeddings
- Built-in embedding functions
- Metadata filtering support

### Qdrant (Port 6333)
High-performance vector similarity search engine.
- Advanced filtering capabilities
- Payload storage alongside vectors
- Production-ready with clustering support

### Weaviate (Port 8081)
AI-native vector database with semantic search capabilities.
- GraphQL-based queries
- Multi-modal data support
- Built-in vectorization modules

## 🔧 Usage Examples

### Python Quick Start

```python
# ChromaDB
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.create_collection("my_collection")
collection.add(
    documents=["This is a document", "This is another document"],
    ids=["id1", "id2"]
)

# Qdrant
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
# See examples folder for more

# Weaviate
import weaviate
client = weaviate.Client("http://localhost:8081")
# See examples folder for more
```

### Full Examples

Check the `examples/` directory for complete working examples:
- `example1_document_rag.py` - Document-based Q&A system
- `example_usage.py` - Basic CRUD operations for all databases

### Test Applications

The `test-applications/` directory contains:
- Performance testing scripts
- Load generation tools
- Comparison benchmarks

## 🎯 Use Cases

- **Development**: Test vector database features without cloud dependencies
- **Prototyping**: Quickly experiment with different vector databases
- **Learning**: Understand how different vector databases work
- **Testing**: Integration testing for applications using vector databases
- **Benchmarking**: Compare performance across different solutions

## 📊 Monitoring

To monitor these databases in production, we recommend using [AURA](https://github.com/jenastar/AURA) - a comprehensive monitoring stack that includes:
- Prometheus metrics collection
- Grafana dashboards
- Real-time performance monitoring
- Query latency tracking

## 🗄️ Data Persistence

All data is persisted in Docker volumes:
- `chromadb_data` - ChromaDB collections and metadata
- `qdrant_data` - Qdrant collections and indices
- `weaviate_data` - Weaviate schemas and objects

To backup your data:
```bash
docker run --rm -v vector-db-stack_chromadb_data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb-backup.tar.gz -C /data .
```

## 🛠️ Configuration

While the stack works out-of-the-box, you can customize settings:

### Environment Variables
Create a `.env` file to override defaults:
```env
# Ports
CHROMADB_PORT=8000
QDRANT_PORT=6333
WEAVIATE_PORT=8081

# Resources
CHROMADB_MEMORY=2g
QDRANT_MEMORY=2g
WEAVIATE_MEMORY=2g
```

### Resource Limits
Edit `docker-compose.yml` to set CPU and memory limits for each service.

## 📝 Management Commands

```bash
# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v

# View logs
docker-compose logs -f [service_name]

# Restart a specific service
docker-compose restart chromadb

# Scale a service (if supported)
docker-compose up -d --scale chromadb=2
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Projects

- [AURA](https://github.com/jenastar/AURA) - Monitoring stack for containerized environments
- Individual database documentation:
  - [ChromaDB Docs](https://docs.trychroma.com/)
  - [Qdrant Docs](https://qdrant.tech/documentation/)
  - [Weaviate Docs](https://weaviate.io/developers/weaviate)