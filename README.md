📊 Monitoring and Metrics
This API exposes a 7set of Prometheus-compatible metrics to monitor performance, usage, and system health. Below is a detailed overview of all exported metrics:

🔄 API-Level Metrics

| Metric Name                   | Type      | Description                                       |
| ----------------------------- | --------- | ------------------------------------------------- |
| `api_request_count`           | `Counter` | Total number of requests received by the API.     |
| `api_request_errors`          | `Counter` | Total number of failed prediction requests.       |
| `api_request_latency_seconds` | `Summary` | Measures the end-to-end latency for API requests. |


⏱️ Inference Metrics

| Metric Name                               | Type        | Description                                                                                                  |
| ----------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| `inference_latency_seconds`               | `Histogram` | Time spent processing individual prediction requests. Helps identify performance bottlenecks.                |
| `model_prediction_counts{gesture=...}`    | `Counter`   | Number of predictions made, labeled by gesture type (e.g., `fist`, `peace`, `swipe_left`).                   |
| `invalid_landmark_errors{error_type=...}` | `Counter`   | Counts malformed or invalid landmark inputs, labeled by the type of error (`count_mismatch`, `empty`, etc.). |


🖥️ System Metrics

| Metric Name         | Type    | Description                                         |
| ------------------- | ------- | --------------------------------------------------- |
| `cpu_usage_percent` | `Gauge` | Real-time CPU usage percentage of the host machine. |
| `ram_usage_percent` | `Gauge` | Real-time RAM usage percentage of the host machine. |


.

🕒 Uptime Metrics

| Metric Name                           | Type    | Description                                        |
| ------------------------------------- | ------- | -------------------------------------------------- |
| `total_app_uptime_seconds`            | `Gauge` | Total uptime in seconds since the API was started. |
| `app_uptime_seconds{status="total"}`  | `Gauge` | Alias for total uptime (for multi-label scraping). |
| `app_uptime_seconds{status="active"}` | `Gauge` | Time since the last request was handled.           |


📡 Metrics Endpoint
Metrics are exposed at:

http://localhost:8000/metrics

✅ Use Cases
Monitor system health (CPU, RAM)

Track prediction usage per gesture

Debug malformed requests

Benchmark model inference latency

Detect unusual API activity or downtime

