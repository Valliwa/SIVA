# SIVA - A Self-Improving Vulnerability Detection Agent

## Author: Valentin Walischewski


## Overview

SIVA is an advanced LLM agent, that uses REVOLVE to dynamically optimize its prompts through memory-guided meta-learning, to improve its vulnerability detection capabilities [2]. SIVA combines sophisticated learning techniques ([2], [3], [4]), with real-world vulnerability data to achieve state-of-the-art performance in security analysis.


## Key Features

- Self-Improvement: Dynamically adapts and improves performance through iterative, memory-guided learning.
- Meta-Learning Architecture: SIVA learns how to learn better [5].
- Real Vulnerability Data: Evaluated on SecVulEval [6], containing thousands of real vulnerabilities from major open-source projects.
- Smart Caching System: Instant analysis of previously seen functions for compute efficiency
- Zero Code Execution: Safe static analysis, without running potentially malicious code.


## Architecture

### Core

1. **Base Agent** (`SIVA.py`)
   - REVOLVE learning framework
   - Smart memory system with instant cache
   - Pattern recognition for CWE types
   - Simple failure count based strategy selection
  
2. **Meta-Learning** (`MetaSIVA.py`)
   - Dynamic prompt library that evolves
   - Strategy weight optimization
   - Failure analysis and adaptation
   - Learning-to-learn capabilities

### Learning Strategies (implemeted through prompt templates)

- Instant Cache: Instant for exact function matches
- Focused Learning: Reuse proven solutions
- Template Transfer: Apply similar CWE patterns
- Multi-Shot Learning: Learn from diverse examples
- CWE-specific: Dynamically evolved templates for specific CWE famlies

### LLM

We used GEMMA3 (27B) in 4-bit qunatization for all our experiments [7]. The (`gemma_server_api.py`) script downloads the model from hugginface and runs it on a GPU. 

#### Alternative LLM Options

1. Use **Mock Mode** - gives simulated LLM responses for testing
2. Use **LLM API** - edit the `SecurityLLMClient` in (`SIVA.py`) for compatibility with your LLM of choice

### Dataset

This implementation uses the SecVulEval dataset [6], consisting of $25,440$ labeled, filtered, and context-enriched C / C++ functions from real-world projects, including critical infrastructure software such as Linux kernel, OpenSSL, and Apache HTTP Server. The dataset includes vulnerable samples, spanning $5,867$ unique CVE's from $145$ different CWE types. 

## Installation

### Requirements

1. Python 3.8+
2. 4GB+ RAM recommended (Memory grows linear and creates substantial overhead)
3. 20-24GB VRAM for Gemma3 27B model + Inference (GPU)
4. Internet connection

### Set Up

```bash
# Clone the repository
git clone https://github.com/yourusername/siva.git
cd siva

# Install dependencies
pip install -r requirements.txt
```
### Dependencies

```txt
# Core SICA-VULN dependencies
pandas>=1.3.0
numpy>=1.21.0
datasets>=2.0.0
transformers>=4.20.0
httpx>=0.24.0

# Gemma3 Server dependencies
fastapi>=0.100.0
uvicorn>=0.23.0
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
pydantic>=2.0.0
```

### LLM Server Setup (Gemma3 27B)


#### Quick Start

1. **Get HuggingFace Token**
   ```bash
   # Sign up at https://huggingface.co
   # Get token from https://huggingface.co/settings/tokens
   
   export HF_TOKEN="your_token_here"
   
   ```

2. **Configure GPU** (Optional)
   ```python
   # Edit gemma_server_api.py line 50
   
   os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Change to your GPU ID
   
   ```

3. **Start the Server**
   ```bash
   python gemma_server_api.py
   ```
   The server will:
   - Download Gemma3 27B (~54GB first time only)
   - Load with 8-bit quantization (~20GB VRAM)
   - Start API server on `http://localhost:8000`

4. **Verify Installation**
   ```bash
   
   # Check health
   curl http://localhost:8000/health
   
   # Test generation
   curl -X POST http://localhost:8000/test
   
   ```

#### Server Features

- **Model**: Gemma3 27B with 128K context window
- **Memory**: ~14GB (4-bit)
- **Enhanced**: Function calling, multimodal ready


## Usage

### Quick Start

```python
# Run the main interface
python Sica_Vuln.py
```

### Available Options

1. **Test Single Vulnerability** - Quick verification (2 minutes)
2. **Quick Benchmark** - 10 samples, 2 iterations (5-10 minutes)
3. **Full Benchmark** - 50 samples, 3 iterations (20-30 minutes)
4. **Balanced CWE Benchmark** - Test across vulnerability types
5. **Show Dataset Statistics** - Explore SecVulEval data
6. **Debug Mode** - Verbose logging for development

### With Meta-Learning

```python
# Run with meta-learning enhancements
python Meta_Sica.py
```

## Project Structure

```
sica-vuln/
├── Sica_Vuln.py              # Main SICA-VULN agent
├── Meta_Sica.py              # Meta-learning enhancements
├── gemma_server_api.py       # Gemma3 27B LLM server
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── sica_vuln_workspace/      # Auto-created workspace
│   ├── cache/                # Dataset cache
│   ├── sica_vuln_memory/     # Learning memory
│   └── meta_prompt_library/  # Evolved prompts
└── hf_cache/                 # Gemma3 model cache (auto-created)
```

## Methodology

### 1. Data Loading
- Downloads SecVulEval dataset from HuggingFace
- Processes real vulnerabilities from open-source projects

### 2. Analysis Pipeline
```
Input Code → Pattern Recognition → Strategy Selection → 
LLM Analysis → Evaluation → Learning → Memory Update
```

### 3. Learning Process
- **Iteration 1**: Baseline analysis
- **Iteration 2**: Apply learned patterns
- **Iteration 3**: Advanced techniques
- **Meta-Learning**: Continuous improvement throughout iterations

### 4. Memory System
- Stores successful analyses
- Builds vulnerability pattern database
- Enables instant cache for similar code functions


## Statement on Generative AI Usage 

Given the implementation heavy and ambitious nature of this project, I have made use of modern software tools in its production process. Namely, I have made extensive use of Claude-4 Sonnet [1] to aid me with the following:

- **Debugging and error fixing**
- **Method prototyping**
- **Improved interpretability** (used it to add detailed logging and debugging logging)
- **Improved robustness** (used it to add edge case protections)
- **Prompt template optimization**
- **Synthetic data and mock response generation**
- **Gemma3 Server implementation** (given the well documented nature of this task I used the LLM to help me create the Gemma3 server script)




## References 

[1] **[Anthropic (2025), Claude-4 Sonnet](https://www.anthropic.com/claude/sonnet)** 

[2] **[Zhang et al. (2025), REVOLVE: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization](https://arxiv.org/abs/2412.03092)**

[3] **[Hu et al. (2025), Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)**

[4] **[Robeyns et al. (2025), A Self-Improving Coding Agent](https://arxiv.org/abs/2504.15228)**

[5] **[T. Liu and M. van der Schaar (2025), Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](https://arxiv.org/abs/2506.05109)**

[6] **[Ahmed et al. (2025), SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection](https://arxiv.org/abs/2505.19828)**

[7] **[Google DeepMind (2024), Gemma3](https://deepmind.google/models/gemma/gemma-3/)**




