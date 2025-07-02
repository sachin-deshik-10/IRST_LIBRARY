# IRST Library - Jupyter Notebook Collection

Welcome to the comprehensive collection of Jupyter notebooks for the IRST Library! These notebooks provide hands-on tutorials, advanced techniques, and complete workflows for infrared small target detection.

## 📚 **Notebook Index**

### 🎯 **Getting Started**

#### 1. [**Complete Tutorial**](irst_tutorial.ipynb) ⭐ **START HERE**

**Complete beginner-to-advanced tutorial covering the entire IRST workflow**

- ✅ **Difficulty**: Beginner to Intermediate
- ⏱️ **Duration**: 2-3 hours
- 🎯 **Topics**: Data loading, training, evaluation, inference, deployment
- 📝 **Prerequisites**: Basic Python and ML knowledge

### 🔬 **Advanced Techniques**

#### 2. [**Advanced Training Techniques**](training_advanced.ipynb)

**Multi-GPU training, hyperparameter optimization, and custom losses**

- ✅ **Difficulty**: Advanced
- ⏱️ **Duration**: 3-4 hours  
- 🎯 **Topics**: Distributed training, mixed precision, custom losses, MLOps
- 📝 **Prerequisites**: Completed basic tutorial, multi-GPU setup

#### 3. [**Model Zoo Exploration**](model_zoo_tutorial.ipynb)

**Compare and select the perfect model for your needs**

- ✅ **Difficulty**: Intermediate
- ⏱️ **Duration**: 1-2 hours
- 🎯 **Topics**: Model comparison, performance analysis, selection criteria
- 📝 **Prerequisites**: Understanding of different architectures

#### 4. [**Dataset Preparation**](dataset_preparation.ipynb)

**Custom datasets and advanced data preparation strategies**

- ✅ **Difficulty**: Intermediate
- ⏱️ **Duration**: 2-3 hours
- 🎯 **Topics**: Custom datasets, augmentation, data quality, annotation
- 📝 **Prerequisites**: Data preprocessing knowledge

### 🚀 **Production & Deployment**

#### 5. [**Production Deployment**](deployment_tutorial.ipynb)

**Deploy models to cloud, edge, and enterprise environments**

- ✅ **Difficulty**: Advanced
- ⏱️ **Duration**: 3-4 hours
- 🎯 **Topics**: Docker, Kubernetes, APIs, monitoring, scaling
- 📝 **Prerequisites**: DevOps knowledge, cloud platform access

#### 6. [**Benchmarking & Analysis**](benchmarking_tutorial.ipynb)

**Comprehensive performance evaluation and optimization**

- ✅ **Difficulty**: Intermediate to Advanced
- ⏱️ **Duration**: 2-3 hours
- 🎯 **Topics**: Performance metrics, statistical analysis, optimization
- 📝 **Prerequisites**: Statistical analysis background

### 🎯 **Complete Workflows**

#### 7. [**End-to-End Workflow**](complete_workflow.ipynb)

**From research to production: complete ISTD pipeline**

- ✅ **Difficulty**: Expert
- ⏱️ **Duration**: 4-6 hours
- 🎯 **Topics**: Complete project lifecycle, best practices, automation
- 📝 **Prerequisites**: All previous notebooks, production experience

## 🛤️ **Learning Paths**

### 🎓 **Beginner Path**

1. **Complete Tutorial** → 2. **Model Zoo** → 3. **Dataset Preparation**

### 🔬 **Researcher Path**  

1. **Complete Tutorial** → 2. **Advanced Training** → 3. **Benchmarking** → 4. **Complete Workflow**

### 🚀 **Engineer Path**

1. **Complete Tutorial** → 2. **Model Zoo** → 3. **Production Deployment** → 4. **Complete Workflow**

### 🎯 **Expert Path**

**All notebooks in sequence for comprehensive mastery**

## 📊 **Quick Reference**

| Notebook | Focus | Difficulty | Duration | GPU Required |
|----------|--------|------------|----------|--------------|
| Complete Tutorial | End-to-end basics | ⭐⭐ | 2-3h | Optional |
| Advanced Training | Multi-GPU & optimization | ⭐⭐⭐⭐ | 3-4h | Recommended |
| Model Zoo | Model comparison | ⭐⭐⭐ | 1-2h | Optional |
| Dataset Preparation | Data workflows | ⭐⭐⭐ | 2-3h | Optional |
| Production Deployment | MLOps & deployment | ⭐⭐⭐⭐ | 3-4h | Optional |
| Benchmarking | Performance analysis | ⭐⭐⭐ | 2-3h | Recommended |
| Complete Workflow | Full pipeline | ⭐⭐⭐⭐⭐ | 4-6h | Recommended |

## 🎯 **Usage Tips**

### 💻 **Environment Setup**

```bash
# Install IRST Library
pip install irst-library

# Install Jupyter
pip install jupyter jupyterlab

# Launch Jupyter Lab
jupyter lab
```

### 📊 **Dataset Downloads**

Most notebooks will automatically download required datasets, but you can pre-download:

```python
from irst_library.datasets import SIRSTDataset
SIRSTDataset(root="./data", download=True)
```

### 🔧 **Hardware Recommendations**

- **Minimum**: 8GB RAM, any GPU with 4GB+ VRAM
- **Recommended**: 16GB+ RAM, RTX 3080+ or V100+
- **Optimal**: 32GB+ RAM, Multiple high-end GPUs

### 🆘 **Getting Help**

- 📖 **Documentation**: [docs/](../docs/)
- 💬 **Discussions**: GitHub Discussions
- 🐛 **Issues**: GitHub Issues
- 📧 **Contact**: <community@irst-library.org>

## 🎉 **Ready to Start?**

Begin with the [**Complete Tutorial**](irst_tutorial.ipynb) for a comprehensive introduction to the IRST Library!

---

*Last updated: July 2, 2025*  
*IRST Library v2.0.0*
