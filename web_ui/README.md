# IRST Library Web UI

## 🌐 Interactive Web Dashboard

Advanced web-based interface for IRST Library with real-time visualization and model management.

### Features

- **Model Training Dashboard** - Real-time training visualization
- **Dataset Explorer** - Interactive dataset browsing and analysis
- **Model Comparison Tool** - Side-by-side model performance comparison
- **Inference Playground** - Upload and test images in real-time
- **Experiment Tracking** - MLflow integration with custom UI
- **Deployment Manager** - One-click model deployment
- **Performance Monitoring** - Real-time system metrics

### Tech Stack

- **Frontend**: React + TypeScript + D3.js
- **Backend**: FastAPI + WebSocket
- **Database**: PostgreSQL + Redis
- **Visualization**: Plotly + Three.js for 3D visualizations

### Setup

```bash
cd web_ui
npm install
npm start  # Frontend
python backend/main.py  # Backend
```
