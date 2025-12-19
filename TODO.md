# 📋 TODO List - Shopping AI Assistant

> تاریخ آخرین بروزرسانی: 19 دسامبر 2025

---

## 🔴 **CRITICAL - فوری (این هفته)**

### امنیت
-  **مخفی کردن IP و اطلاعات حساس Elasticsearch**
  - جابجایی به `.env`
  - استفاده از Private Network یا VPN
  - محدودسازی دسترسی با Firewall
  - فایل: `config/.env.example`

-  **پیاده‌سازی Authentication برای MCP Servers**
  - اضافه کردن API Key middleware
  - JWT Token برای Agent
  - فایل‌ها: `src/mcp_servers/*.py`

-  **اضافه کردن Rate Limiting**
  - استفاده از `slowapi`
  - 10 request/minute per IP
  - فایل‌ها: `src/mcp_servers/*.py`

### Pipeline Performance
-  **اضافه کردن Redis Cache برای Embeddings**
  ```python
  # src/mcp_servers/embedding_server.py
  - استفاده از Redis
  - TTL: 3600 seconds
  - Cache key: f"emb:{hash(text)}"
  ```
  - فایل: `src/mcp_servers/embedding_server.py`

-  **کاهش Timeout EQuIP از 120s به 10s**
  - فایل: `src/mcp_servers/equip_server.py`, line 108
  - تغییر `timeout=120` به `timeout=10`

-  **Fix Token Mapping Fallback**
  ```python
  # src/mcp_servers/interpret_server.py
  if not token_mapping or len(token_mapping) == 0:
      token_mapping = auto_extract_tokens(equip_prompt, persian_full_query)
  ```
  - فایل: `src/mcp_servers/interpret_server.py`

-  **Fix Category Filter - اطمینان از اضافه شدن**
  - فایل: `src/mcp_servers/dsl_processor_server.py`
  - متد: `_fix_category_filters` → تغییر به `_ensure_category_filters`

### Testing
-  **ایجاد Unit Tests**
  ```bash
  tests/
    unit/
      test_interpret_service.py
      test_search_service.py
      test_equip_service.py
      test_dsl_processor.py
      test_embedding_service.py
  ```

-  **ایجاد Integration Tests**
  ```bash
  tests/
    integration/
      test_full_pipeline.py
      test_agent_flow.py
  ```

-  **اضافه کردن pytest و coverage**
  ```bash
  pip install pytest pytest-asyncio pytest-cov
  pytest tests/ --cov=src --cov-report=html
  ```

---

## 🟡 **MAJOR - کوتاه‌مدت (1-2 هفته)**

### Architecture
-  **Refactor SearchService - تقسیم به کلاس‌های کوچک‌تر**
  ```python
  # ایجاد:
  src/services/
    embedding_client.py
    elasticsearch_client.py
    mcp_client.py
    value_score_calculator.py
    search_orchestrator.py
  ```

-  **پیاده‌سازی Retry Logic با Tenacity**
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential
  
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
  async def call_mcp_tool_with_retry(...):
      ...
  ```
  - فایل‌ها: `src/agent.py`, `src/mcp_servers/search_server.py`

-  **پیاده‌سازی Circuit Breaker**
  ```python
  from pybreaker import CircuitBreaker
  
  breaker = CircuitBreaker(fail_max=5, timeout_duration=60)
  
  @breaker
  async def call_equip_server(...):
      ...
  ```

-  **اضافه کردن Health Check Endpoints**
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
  - فایل‌ها: همه `src/mcp_servers/*.py`

### Pipeline Optimization
-  **Parallel Processing در Interpret Stage**
  ```python
  # src/mcp_servers/interpret_server.py
  interpret_task, category_task = await asyncio.gather(
      self.llm.invoke(...),
      self.classify_categories(...)
  )
  ```

-  **Graceful Degradation برای EQuIP**
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

-  **بهبود Score Normalization**
  ```python
  # src/mcp_servers/search_server.py
  # استفاده از Min-Max Normalization به جای /5.0
  scores = [hit['_score'] for hit in hits]
  normalized = (score - min(scores)) / (max(scores) - min(scores))
  ```

-  **Query Result Caching با Redis**
  ```python
  cache_key = f"search:{query_hash}:{categories}"
  cached = await redis.get(cache_key)
  if cached:
      return json.loads(cached)
  ```

### Configuration
-  **استفاده از Pydantic Settings**
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

-  **ایجاد config.yaml**
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
-  **ایجاد Docker Compose**
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

-  **ایجاد Dockerfile**
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["python", "main.py"]
  ```

-  **Setup CI/CD با GitHub Actions**
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

## 🟢 **MINOR - میان‌مدت (1 ماه)**

### Code Quality
-  **اضافه کردن Type Hints کامل**
  - همه فایل‌های `src/`
  - استفاده از `mypy` برای type checking

-  **Fix Duplicate Code**
  - تابع `call_mcp_tool` در 3 فایل تکرار شده
  - ایجاد `src/utils/mcp_client.py`

-  **حذف Magic Numbers**
  ```python
  # قبل:
  similarity = min(1.0, score / 5.0)
  
  # بعد:
  SIMILARITY_NORMALIZATION_FACTOR = 5.0
  similarity = min(1.0, score / SIMILARITY_NORMALIZATION_FACTOR)
  ```

-  **بهبود Error Messages**
  ```python
  # قبل:
  except Exception as e:
      print(f"خطا: {str(e)}")
  
  # بعد:
  except ConnectionError as e:
      logger.error("خطا در اتصال به سرور جستجو. لطفاً دوباره تلاش کنید.")
  except ValueError as e:
      logger.error("ورودی نامعتبر است. لطفاً جستجوی خود را اصلاح کنید.")
  ```

### Monitoring & Observability
-  **اضافه کردن Prometheus Metrics**
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

-  **بهبود Structured Logging**
  ```python
  logger.info("Search completed", extra={
      "query": query,
      "results_count": len(results),
      "latency_ms": latency,
      "cache_hit": cache_hit
  })
  ```

### Performance
-  **پیاده‌سازی Connection Pooling**
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

-  **Database Query Optimization**
  - بررسی ES query performance
  - اضافه کردن indexes مناسب

### Data Validation
-  **استفاده از Pydantic Models**
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

-  **JSON Schema Validation برای DSL**
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
-  **API Documentation با Swagger/OpenAPI**
  - استفاده از FastAPI's built-in docs
  - مستندسازی تمام endpoints

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

-  **اضافه کردن کامنت‌های Docstring**
  - همه توابع public
  - format: Google style

---

## 🔵 **NICE TO HAVE - بلندمدت (3+ ماه)**

### Advanced Features
-  **پیاده‌سازی API Gateway**
  - Kong یا Nginx
  - Centralized authentication
  - Load balancing

-  **Service Discovery**
  - Consul یا etcd
  - Dynamic service registration

-  **Event-Driven Architecture**
  - RabbitMQ یا Kafka
  - Async message processing

-  **CQRS Pattern**
  - جداسازی Command و Query
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
-  **A/B Testing Framework**
  - تست فرمول‌های مختلف value_score
  - تست مدل‌های embedding مختلف

-  **Model Monitoring**
  - Embedding drift detection
  - Query quality metrics

-  **Fine-tune Models**
  - Fine-tune embedding model روی داده‌های فارسی
  - Fine-tune EQuIP روی DSL های واقعی

### User Features
-  **Multi-tenancy Support**
  - جداسازی داده‌های کاربران
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

## 📊 **پیشرفت کلی**

```
امنیت:           [░░░░░░░░░░] 0%   (0/3 تکمیل)
Performance:     [░░░░░░░░░░] 0%   (0/4 تکمیل)
Testing:         [░░░░░░░░░░] 0%   (0/3 تکمیل)
Architecture:    [░░░░░░░░░░] 0%   (0/4 تکمیل)
Pipeline:        [░░░░░░░░░░] 0%   (0/4 تکمیل)
Configuration:   [░░░░░░░░░░] 0%   (0/2 تکمیل)
DevOps:          [░░░░░░░░░░] 0%   (0/3 تکمیل)
Code Quality:    [░░░░░░░░░░] 0%   (0/4 تکمیل)
Monitoring:      [░░░░░░░░░░] 0%   (0/3 تکمیل)
Documentation:   [░░░░░░░░░░] 0%   (0/4 تکمیل)

کل پیشرفت:      [░░░░░░░░░░] 0%   (0/38 تکمیل)
```

---

## 🎯 **اولویت‌بندی هفته به هفته**

### هفته 1
-  Redis Cache
-  Fix Timeout
-  Token Mapping Fallback
-  Category Filter Fix
-  Unit Tests (basic)

### هفته 2
-  Authentication
-  Rate Limiting
-  Health Checks
-  Retry Logic
-  Integration Tests

### هفته 3
-  Refactor SearchService
-  Parallel Processing
-  Graceful Degradation
-  Docker Compose

### هفته 4
-  CI/CD Setup
-  Pydantic Settings
-  Connection Pooling
-  Prometheus Metrics

---

## 📝 **نکات مهم**

### برای شروع سریع:
```bash
# 1. نصب dependencies جدید
pip install redis tenacity pybreaker pydantic-settings pytest pytest-asyncio

# 2. راه‌اندازی Redis
docker run -d -p 6379:6379 redis:alpine

# 3. اجرای تست‌ها
pytest tests/ -v

# 4. چک کردن coverage
pytest tests/ --cov=src --cov-report=html
```

### چک‌لیست قبل از Production:
-  تمام تست‌ها Pass می‌شوند
-  Coverage بالای 80%
-  Security scan (bandit, safety)
-  Load testing (Locust)
-  Documentation کامل است
-  Monitoring setup شده
-  Backup strategy تعریف شده
-  Rollback plan آماده است

---

## 🔗 **منابع مفید**

- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Redis Caching Guide](https://redis.io/docs/manual/patterns/caching/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/usage/settings/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

---

**یادآوری:** این TODO list زنده است و باید با پیشرفت پروژه به‌روزرسانی شود.

```bash
# برای آپدیت کردن پیشرفت:
# هر وقت یک task تکمیل شد، علامت [x] بزنید
# مثال: - [x] Task completed
```
