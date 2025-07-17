# Vector Database Monitoring Stack

A standalone monitoring solution for vector databases including ChromaDB, Pinecone, FAISS, and others.

## Features

- Real-time metrics collection from vector databases
- Prometheus-based metric storage
- Grafana dashboards for visualization
- Support for multiple vector database types
- Docker-based deployment

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd vector-db-monitor
```

2. Configure environment variables:
```bash
cp config/vector-db-config.env.example config/vector-db-config.env
# Edit config/vector-db-config.env with your settings
```

3. Start the stack:
```bash
docker-compose up -d
```

4. Access services:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- ChromaDB: http://localhost:8000
- Vector DB Exporter: http://localhost:9402/metrics

## Architecture

```
├── docker-compose.yml          # Main compose file
├── exporters/                  # Metric exporters
│   └── vector-db/             # Vector DB exporter
├── prometheus/                 # Prometheus configuration
├── grafana/                    # Grafana dashboards and config
├── config/                     # Configuration files
├── test-applications/          # Test and demo applications
└── examples/                   # Usage examples
```

## Supported Vector Databases

- ChromaDB
- Pinecone
- FAISS
- Qdrant (planned)
- Weaviate (planned)

## Configuration

Edit `config/vector-db-config.env` to set:
- Database connection parameters
- API keys for cloud services
- Metric collection intervals
- Custom labels and tags

## Testing

Run test applications:
```bash
cd test-applications
python test_vector_operations.py
```

## Development

To add support for a new vector database:
1. Create a wrapper in `exporters/vector-db/`
2. Add metrics collection logic
3. Update the exporter configuration
4. Create test cases

## Monitoring Metrics

Key metrics collected:
- Query latency and throughput
- Index size and document count
- Memory and CPU usage
- Error rates and types
- Collection statistics

## License

MIT