# 📋 TODO List - Shopping AI Assistant

> Last Updated: December 19, 2025

---

## 🔴 **CRITICAL - Urgent (This Week)**

### Security
- [ ] **Hide Elasticsearch IP and sensitive credentials**
  - Move to `.env` file
  - Use Private Network or VPN
  - Restrict access with Firewall
  - File: `config/.env.example`

- [ ] **Implement Authentication for MCP Servers**
  - Add API Key middleware
  - JWT Token for Agent
  - Files: `src/mcp_servers/*.py`

- [ ] **Add Rate Limiting**
  - Use `slowapi`
  - 10 requests/minute per IP
  - Files: `src/mcp_servers/*.py`

### Pipeline Performance
- [ ] **Add Redis Cache for Embeddings**
  ```python
  # src/mcp_servers/embedding_server.py
  - Use Redis
  - TTL: 3600 seconds
  - Cache key: f"emb:{hash(text)}"
  ```
  - File: `src/mcp_servers/embedding_server.py`

- [ ] **Reduce EQuIP Timeout from 120s to 10s**
  - File: `src/mcp_servers/equip_server.py`, line 108
  - Change `timeout=120` to `timeout=10`

-  **Fix Token Mapping Fallback**
  ```python
  # src/mcp_servers/interpret_server.py
  if not token_mapping or len(token_mapping) == 0:
      token_mapping = auto_extract_tokens(equip_prompt, persian_full_query)
  ```
  - File: `src/mcp_servers/interpret_server.py`

- [ ] **Fix Category Filter - Ensure it's added**
  - File: `src/mcp_servers/dsl_processor_server.py`
  - Method: `_fix_category_filters` → Change to `_ensure_category_filters`

### Testing
- [ ] **Create Unit Tests**
  ```bash
  tests/
    unit/
      test_interpret_service.py
      test_search_service.py
      test_equip_service.py
      test_dsl_processor.py
      test_embedding_service.py
  ```

- [ ] **Create Integration Tests**
  ```bash
  tests/
    integration/
      test_full_pipeline.py
      test_agent_flow.py
  ```

- [ ] **Add pytest and coverage**
  ```bash
  pip install pytest pytest-asyncio pytest-cov
  pytest tests/ --cov=src --cov-report=html
  ```

---

## 🟡 **MAJOR - Short-term (1-2 Weeks)**

### Architecture
- [ ] **Refactor SearchService - Split into smaller classes**
  ```python
  # Create:
  src/services/
    embedding_client.py
    elasticsearch_client.py
    mcp_client.py
    value_score_calculator.py
    search_orchestrator.py
  ```

- [ ] **Implement Retry Logic with Tenacity**
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential
  
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
  async def call_mcp_tool_with_retry(...):
      ...
  ```
  - Files: `src/agent.py`, `src/mcp_servers/search_server.py`

- [ ] **Implement Circuit Breaker**
  ```python
  from pybreaker import CircuitBreaker
  
  breaker = CircuitBreaker(fail_max=5, timeout_duration=60)
  
  @breaker
  async def call_equip_server(...):
      ...
  ```

- [ ] **Add Health Check Endpoints**
  ```python
  @app.get("/health")
  async def health():
      return {
          "status": "healthy",
          "timestamp": time.time(),
          "dependencies": {
              "elasticsearch": check_es(),
              "embedding_model": check_model()
          }
      }
  ```
  - Files: All `src/mcp_servers/*.py`

### Pipeline Optimization
- [ ] **Parallel Processing in Interpret Stage**
  ```python
  # src/mcp_servers/interpret_server.py
  interpret_task, category_task = await asyncio.gather(
      self.llm.invoke(...),
      self.classify_categories(...)
  )
  ```

- [ ] **Graceful Degradation for EQuIP**
  ```python
  # src/mcp_servers/search_server.py
  try:
      dsl = await asyncio.wait_for(
          self.call_equip_server(...),
          timeout=5.0
      )
  except:
      logger.warning("EQuIP failed, using template DSL")
      dsl = create_simple_dsl(...)
  ```

- [ ] **Improve Score Normalization**
  ```python
  # src/mcp_servers/search_server.py
  # Use Min-Max Normalization instead of /5.0
  scores = [hit['_score'] for hit in hits]
  normalized = (score - min(scores)) / (max(scores) - min(scores))
  ```

- [ ] **Query Result Caching with Redis**
  ```python
  cache_key = f"search:{query_hash}:{categories}"
  cached = await redis.get(cache_key)
  if cached:
      return json.loads(cached)
  ```

### Configuration
- [ ] **Use Pydantic Settings**
  ```python
  # src/config/settings.py
  from pydantic_settings import BaseSettings
  
  class Settings(BaseSettings):
      debug_mode: bool = False
      server_port: int
      elasticsearch_host: str
      
      class Config:
          env_file = '.env'
  ```

- [ ] **Create config.yaml**
  ```yaml
  servers:
    embedding:
      port: 5003
      timeout: 30
    interpret:
      port: 5004
      timeout: 10
  ```

### DevOps
- [ ] **Create Docker Compose**
  ```yaml
  # docker-compose.yml
  services:
    redis:
      image: redis:alpine
    elasticsearch:
      image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    embedding-server:
      build: .
      command: python src/mcp_servers/embedding_server.py
  ```

- [ ] **Create Dockerfile**
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "main.py"]
  ```

- [ ] **Setup CI/CD with GitHub Actions**
  ```yaml
  # .github/workflows/test.yml
  name: Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        - name: Run tests
          run: pytest tests/
  ```

---

## 🟢 **MINOR - Mid-term (1 Month)**

### Code Quality
- [ ] **Add Complete Type Hints**
  - All files in `src/`
  - Use `mypy` for type checking

- [ ] **Fix Duplicate Code**
  - `call_mcp_tool` function duplicated in 3 files
  - Create `src/utils/mcp_client.py`

- [ ] **Remove Magic Numbers**
  ```python
  # Before:
  similarity = min(1.0, score / 5.0)
  
  # After:
  SIMILARITY_NORMALIZATION_FACTOR = 5.0
  similarity = min(1.0, score / SIMILARITY_NORMALIZATION_FACTOR)
  ```

- [ ] **Improve Error Messages**
  ```python
  # Before:
  except Exception as e:
      print(f"Error: {str(e)}")
  
  # After:
  except ConnectionError as e:
      logger.error("Connection error to search server. Please try again.")
  except ValueError as e:
      logger.error("Invalid input. Please correct your query.")
  ```

### Monitoring & Observability
- [ ] **Add Prometheus Metrics**
  ```python
  from prometheus_client import Counter, Histogram
  
  search_requests = Counter('search_requests_total', 'Total search requests')
  search_latency = Histogram('search_duration_seconds', 'Search latency')
  
  @search_latency.time()
  async def search(...):
      search_requests.inc()
      ...
  ```

-  **Setup Grafana Dashboard**
  - Query Rate
  - Latency (p50, p95, p99)
  - Error Rate
  - Cache Hit Rate

- [ ] **Improve Structured Logging**
  ```python
  logger.info("Search completed", extra={
      "query": query,
      "results_count": len(results),
      "latency_ms": latency,
      "cache_hit": cache_hit
  })
  ```

### Performance
- [ ] **Implement Connection Pooling**
  ```python
  # src/mcp_servers/search_server.py
  app.state.http_client = aiohttp.ClientSession()  # Reuse
  ```

-  **Batch Embedding Processing**
  ```python
  embeddings = await get_embeddings_batch([
      persian_full_query,
      *categories
  ])
  ```

- [ ] **Database Query Optimization**
  - Review ES query performance
  - Add appropriate indexes

### Data Validation
- [ ] **Use Pydantic Models**
  ```python
  # src/models/search.py
  from pydantic import BaseModel, validator
  
  class SearchRequest(BaseModel):
      equip_prompt: str
      price_sensitivity: float
      
      @validator('equip_prompt')
      def validate_prompt(cls, v):
          if len(v) < 3:
              raise ValueError('Prompt too short')
          return v
  ```

- [ ] **JSON Schema Validation for DSL**
  ```python
  from jsonschema import validate
  
  DSL_SCHEMA = {
      "type": "object",
      "properties": {
          "query": {"type": "object"},
          "size": {"type": "integer"}
      }
  }
  ```

### Documentation
- [ ] **API Documentation with Swagger/OpenAPI**
  - Use FastAPI's built-in docs
  - Document all endpoints

-  **Architecture Decision Records (ADR)**
  ```markdown
  docs/adr/
    0001-use-mcp-protocol.md
    0002-separate-embedding-server.md
    0003-use-equip-for-dsl.md
  ```

-  **Troubleshooting Guide**
  ```markdown
  docs/troubleshooting.md
  - EQuIP connection issues
  - Elasticsearch timeout
  - Empty results
  ```

- [ ] **Add Docstring Comments**
  - All public functions
  - Format: Google style

---

## 🔵 **NICE TO HAVE - Long-term (3+ Months)**

### Advanced Features
- [ ] **Implement API Gateway**
  - Kong or Nginx
  - Centralized authentication
  - Load balancing

- [ ] **Service Discovery**
  - Consul or etcd
  - Dynamic service registration

- [ ] **Event-Driven Architecture**
  - RabbitMQ or Kafka
  - Async message processing

- [ ] **CQRS Pattern**
  - Separate Command and Query
  - Event Sourcing

### Scalability
-  **Kubernetes Deployment**
  ```yaml
  # k8s/deployment.yml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: embedding-server
  spec:
    replicas: 3
  ```

-  **Horizontal Pod Autoscaling**
  ```yaml
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: embedding-server-hpa
  spec:
    minReplicas: 2
    maxReplicas: 10
    metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  ```

-  **Multi-region Deployment**
  - Active-Active setup
  - Global Load Balancer

### ML/AI
- [ ] **A/B Testing Framework**
  - Test different value_score formulas
  - Test different embedding models

-  **Model Monitoring**
  - Embedding drift detection
  - Query quality metrics

- [ ] **Fine-tune Models**
  - Fine-tune embedding model on Persian data
  - Fine-tune EQuIP on real DSL queries

### User Features
- [ ] **Multi-tenancy Support**
  - Isolate user data
  - Tenant-specific configs

-  **Personalization**
  - User preferences
  - Search history
  - Recommendations

-  **Advanced Filters**
  - Price range slider
  - Brand selection
  - Color/Size filters

---

## 📊 **Overall Progress**

```
Security:         [░░░░░░░░░░] 0%   (0/3 completed)
Performance:      [░░░░░░░░░░] 0%   (0/4 completed)
Testing:          [░░░░░░░░░░] 0%   (0/3 completed)
Architecture:     [░░░░░░░░░░] 0%   (0/4 completed)
Pipeline:         [░░░░░░░░░░] 0%   (0/4 completed)
Configuration:    [░░░░░░░░░░] 0%   (0/2 completed)
DevOps:           [░░░░░░░░░░] 0%   (0/3 completed)
Code Quality:     [░░░░░░░░░░] 0%   (0/4 completed)
Monitoring:       [░░░░░░░░░░] 0%   (0/3 completed)
Documentation:    [░░░░░░░░░░] 0%   (0/4 completed)

Total Progress:   [░░░░░░░░░░] 0%   (0/38 completed)
```

---

## 🎯 **Week-by-Week Priorities**

### Week 1
- [ ] Redis Cache
- [ ] Fix Timeout
- [ ] Token Mapping Fallback
- [ ] Category Filter Fix
- [ ] Unit Tests (basic)

### Week 2
- [ ] Authentication
- [ ] Rate Limiting
- [ ] Health Checks
- [ ] Retry Logic
- [ ] Integration Tests

### Week 3
- [ ] Refactor SearchService
- [ ] Parallel Processing
- [ ] Graceful Degradation
- [ ] Docker Compose

### Week 4
- [ ] CI/CD Setup
- [ ] Pydantic Settings
- [ ] Connection Pooling
- [ ] Prometheus Metrics

---

## 📝 **Important Notes**

### Quick Start:
```bash
# 1. Install new dependencies
pip install redis tenacity pybreaker pydantic-settings pytest pytest-asyncio

# 2. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 3. Run tests
pytest tests/ -v

# 4. Check coverage
pytest tests/ --cov=src --cov-report=html
```

### Pre-Production Checklist:
- [ ] All tests pass
- [ ] Coverage above 80%
- [ ] Security scan (bandit, safety)
- [ ] Load testing (Locust)
- [ ] Documentation complete
- [ ] Monitoring setup
- [ ] Backup strategy defined
- [ ] Rollback plan ready

---

## 🔗 **Useful Resources**

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Redis Caching Guide](https://redis.io/docs/manual/patterns/caching/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/usage/settings/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

---

**Reminder:** This TODO list is a living document and should be updated as the project progresses.

```bash
# To update progress:
# When a task is completed, mark it with [x]
# Example: - [x] Task completed
```
