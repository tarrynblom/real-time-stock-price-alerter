# 🚀 Deployment & Troubleshooting Guide

Comprehensive guide for deploying the Stock Price Alerter and resolving common issues.

## 📋 Table of Contents

- [Quick Deployment](#quick-deployment)
- [Environment Setup](#environment-setup)
- [Common Issues](#common-issues)
- [System Requirements](#system-requirements)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Performance Optimization](#performance-optimization)
- [Production Considerations](#production-considerations)

---

## 🚀 Quick Deployment

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/tarryn-blom/real-time-stock-price-alerter.git
cd real-time-stock-price-alerter

# 2. Set up environment
cp env.template .env
# Edit .env with your API keys

# 3. Deploy with script
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# 4. Verify deployment
curl http://localhost:8000/health
```

### Option 2: Manual Setup

```bash
# 1. Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up configuration
cp env.template .env
# Edit .env file

# 4. Create logs directory
mkdir -p logs

# 5. Start the application
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Option 3: Development Mode

```bash
# For development with auto-reload
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## ⚙️ Environment Setup

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
ALPHA_VANTAGE_API_KEY=your_api_key_here
FINANCIAL_MODELING_PREP_API_KEY=optional_secondary_api_key

# Server Configuration
API_PORT=8000
LOG_LEVEL=info

# Optional: Advanced Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
DEFAULT_STOCK_SYMBOL=AAPL
PREDICTION_THRESHOLD=0.01
CACHE_TTL=300
```

### API Key Setup

#### Alpha Vantage (Primary)
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for free API key
3. Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key_here`

**Free Tier Limits:**
- 5 API calls per minute
- 500 API calls per day

#### Financial Modeling Prep (Optional)
1. Visit [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
2. Sign up for API access
3. Add to `.env`: `FINANCIAL_MODELING_PREP_API_KEY=your_key_here`

---

## 🔧 Common Issues

### Issue 1: API Not Starting

**Symptoms:**
```bash
ERROR: Failed to start API server
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

```bash
# Solution A: Fix Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Solution B: Run from project root
cd /path/to/real-time-stock-price-alerter
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Solution C: Use relative imports
python -c "import sys; print(sys.path)"  # Verify project is in path
```

### Issue 2: Port Already in Use

**Symptoms:**
```bash
OSError: [Errno 48] Address already in use
```

**Solutions:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Or use different port
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Update environment variable
echo "API_PORT=8001" >> .env
```

### Issue 3: Missing Dependencies

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'pandas'
```

**Solutions:**

```bash
# Verify virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For specific missing packages
pip install fastapi uvicorn pandas scikit-learn loguru

# Verify installation
pip list | grep fastapi
python -c "import fastapi; print(fastapi.__version__)"
```

### Issue 4: API Key Issues

**Symptoms:**
```bash
ERROR: API Error: Invalid API call. Please retry or visit the documentation
ERROR: Failed to fetch data for AAPL
```

**Solutions:**

```bash
# Check .env file exists and has correct format
ls -la .env
cat .env | grep ALPHA_VANTAGE_API_KEY

# Verify API key is valid
curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey=YOUR_API_KEY"

# Test with demo key (limited functionality)
echo "ALPHA_VANTAGE_API_KEY=demo" > .env

# Check rate limiting
# Alpha Vantage: 5 calls/minute, 500 calls/day
```

### Issue 5: Docker Issues

**Symptoms:**
```bash
Cannot connect to the Docker daemon
docker: Error response from daemon
```

**Solutions:**

```bash
# Start Docker service
sudo systemctl start docker  # Linux
open -a Docker             # macOS

# Check Docker status
docker --version
docker ps

# Rebuild container if needed
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check container logs
docker-compose logs api
docker-compose logs -f api  # Follow logs
```

### Issue 6: Health Check Failures

**Symptoms:**
```bash
curl: (7) Failed to connect to localhost port 8000
{"detail": "Internal Server Error"}
```

**Diagnostic Steps:**

```bash
# 1. Check if API is running
curl -v http://localhost:8000/health

# 2. Check application logs
tail -f logs/stock_alerter.log

# 3. Test with verbose logging
LOG_LEVEL=debug python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 4. Test individual components
python -c "
from src.core.data_ingestion import DataIngestionService
service = DataIngestionService()
result = service.fetch_stock_data('AAPL')
print('Data ingestion:', result is not None)
"
```

### Issue 7: Model Training Failures

**Symptoms:**
```bash
ERROR: Feature validation failed
ERROR: Insufficient data: need at least 5 periods
```

**Solutions:**

```bash
# Check data availability
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "interval": "5min"}' -v

# Verify with known good symbol
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "IBM", "interval": "5min"}'

# Check API rate limits
# Wait 15+ seconds between training requests

# Manual validation
python -c "
from src.core.data_ingestion import DataIngestionService
from src.core.data_preprocessing import DataPreprocessor
service = DataIngestionService()
preprocessor = DataPreprocessor()

dataset = service.fetch_stock_data('AAPL')
if dataset:
    df = preprocessor.preprocess_dataset(dataset)
    print(f'Processed data shape: {df.shape if df is not None else \"None\"}')
else:
    print('Failed to fetch data')
"
```

### Issue 8: Memory Issues

**Symptoms:**
```bash
MemoryError: Unable to allocate array
Process killed (out of memory)
```

**Solutions:**

```bash
# Check memory usage
free -h  # Linux
top -l 1 | grep PhysMem  # macOS

# Reduce dataset size
# Edit src/core/data_preprocessing.py
# Limit data points in preprocessing

# Increase Docker memory limit
# Edit docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G

# Monitor memory usage
docker stats stock-alerter-api
```

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 512MB available
- **Storage**: 1GB free space
- **Network**: Internet connection for API access

### Recommended Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.11
- **RAM**: 2GB available
- **Storage**: 5GB free space
- **CPU**: 2+ cores for better performance

### Production Requirements
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: 3.11
- **RAM**: 4GB+ available
- **Storage**: 10GB+ free space
- **CPU**: 4+ cores
- **Network**: High-speed internet, low latency

---

## 📊 Monitoring & Health Checks

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/metrics

# Expected healthy response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_trained": true,
  "version": "1.0.0",
  "services": {
    "api": {"status": "healthy"}
  },
  "metrics": {
    "models_loaded": 1
  }
}
```

### Log Monitoring

```bash
# Monitor application logs
tail -f logs/stock_alerter.log

# Monitor Docker logs
docker-compose logs -f api

# Search for errors
grep -i error logs/stock_alerter.log
grep -i "failed" logs/stock_alerter.log

# Monitor alerts
tail -f logs/alerts.log
```

### Performance Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Basic performance monitoring

echo "=== Stock Alerter Health Monitor ==="
echo "Timestamp: $(date)"
echo

# API Health Check
echo "🔍 API Health Check:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding"
    HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
    echo "Status: $HEALTH"
else
    echo "❌ API is not responding"
fi

# Docker Status
echo -e "\n🐳 Docker Status:"
if docker ps | grep -q stock-alerter-api; then
    echo "✅ Container is running"
    docker stats stock-alerter-api --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
else
    echo "❌ Container is not running"
fi

# Log Analysis
echo -e "\n📋 Recent Errors:"
tail -n 20 logs/stock_alerter.log | grep -i error | tail -n 5

echo -e "\n📈 Recent Alerts:"
if [ -f logs/alerts.log ]; then
    tail -n 5 logs/alerts.log
else
    echo "No alerts logged"
fi

echo -e "\n=== End Monitor ==="
```

Usage:
```bash
chmod +x monitor.sh
./monitor.sh

# Run continuously
watch -n 30 ./monitor.sh
```

---

## ⚡ Performance Optimization

### API Performance Tuning

```bash
# Increase worker processes for production
python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker

# Or using gunicorn
pip install gunicorn
gunicorn src.api.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Memory Optimization

```python
# Add to src/core/data_preprocessing.py
class DataPreprocessor:
    def __init__(self, max_data_points=100):
        self.max_data_points = max_data_points
    
    def preprocess_dataset(self, dataset):
        df = dataset.to_dataframe()
        
        # Limit data size for memory efficiency
        if len(df) > self.max_data_points:
            df = df.tail(self.max_data_points)
        
        # Continue with existing preprocessing...
```

### Database Caching (Optional)

```python
# Add Redis caching for better performance
import redis
import json
from datetime import timedelta

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
    
    def cache_prediction(self, symbol, prediction, ttl_minutes=5):
        key = f"prediction:{symbol}"
        self.redis_client.setex(
            key, 
            timedelta(minutes=ttl_minutes), 
            json.dumps(prediction)
        )
    
    def get_cached_prediction(self, symbol):
        key = f"prediction:{symbol}"
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
```

---

## 🏭 Production Considerations

### Security Hardening

```bash
# 1. Environment variables
export ALPHA_VANTAGE_API_KEY="your_secret_key"
export LOG_LEVEL="warning"  # Reduce log verbosity

# 2. Firewall rules
sudo ufw allow 8000/tcp  # Only allow API port

# 3. Reverse proxy (nginx)
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL/HTTPS Setup

```yaml
# docker-compose.yml for production
services:
  api:
    build: .
    environment:
      - SSL_CERT_PATH=/etc/ssl/certs/cert.pem
      - SSL_KEY_PATH=/etc/ssl/private/key.pem
    volumes:
      - ./ssl:/etc/ssl
    ports:
      - "443:8000"
```

### Process Management

```bash
# Using systemd service
sudo tee /etc/systemd/system/stock-alerter.service > /dev/null <<EOF
[Unit]
Description=Stock Price Alerter API
After=network.target

[Service]
Type=exec
User=stockalerter
WorkingDirectory=/opt/stock-alerter
Environment=PATH=/opt/stock-alerter/venv/bin
ExecStart=/opt/stock-alerter/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable stock-alerter
sudo systemctl start stock-alerter
sudo systemctl status stock-alerter
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Backup logs and configuration

BACKUP_DIR="/backup/stock-alerter/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration
cp .env $BACKUP_DIR/
cp docker-compose.yml $BACKUP_DIR/

# Backup logs
cp -r logs/ $BACKUP_DIR/

# Backup any trained models (if saved)
if [ -d "models/" ]; then
    cp -r models/ $BACKUP_DIR/
fi

echo "Backup completed: $BACKUP_DIR"
```

---

## 🆘 Emergency Procedures

### Quick Recovery Steps

```bash
# 1. Stop everything
docker-compose down
pkill -f uvicorn

# 2. Clean restart
docker system prune -f
rm -rf logs/*

# 3. Fresh deployment
git pull origin main
./scripts/deploy.sh

# 4. Verify functionality
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"symbol": "AAPL"}'
```

### Data Recovery

```bash
# If logs are corrupted
mkdir -p logs
touch logs/stock_alerter.log
touch logs/alerts.log
chmod 666 logs/*.log

# If configuration is lost
cp env.template .env
# Manually add API keys
```

### Contact Information

**For Production Issues:**
- Check logs: `tail -f logs/stock_alerter.log`
- Check health: `curl http://localhost:8000/health`
- Restart service: `docker-compose restart api`

**For Development Support:**
- GitHub Issues: [Create Issue](https://github.com/your-username/real-time-stock-price-alerter/issues)
- Documentation: [API Docs](http://localhost:8000/docs)

---

**💡 Pro Tip**: Set up monitoring alerts for production deployments using tools like Grafana, Prometheus, or simple cron-based health checks that notify you when the service is down. 