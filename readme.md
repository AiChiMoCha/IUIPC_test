# Privacy Attitude Evaluation for Language Models

# 🚧 Under Construction 🚧

This repository evaluates privacy attitudes of language models using standardized Internet Users' Information Privacy Concerns (IUIPC) questionnaires and 7-point Likert scales.

## 📖 Overview

The system evaluates language models across three privacy dimensions:
- **Control**: Desire to control personal data collection and use
- **Awareness**: Need to be informed about data practices  
- **Collection**: Attitudes toward data collection (reverse-scored)

**Privacy Score (temporary)** = (Control + Awareness + (8 - Collection)) / 3

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/AiChiMoCha/IUIPC_test.git
cd IUIPC_test
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# For AWS Bedrock models
export AWS_ACCESS_KEY_ID=your_aws_access_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
export AWS_DEFAULT_REGION=us-west-2

# For OpenAI API models
export OPENAI_API_KEY=your_openai_api_key

# Optional: For local generative models
export MISTRAL_MODEL_PATH=path/to/mistral/model
export LLAMA_MODEL_PATH=path/to/llama/model
...
```

## 🚀 Usage

### AWS Bedrock Evaluation
Evaluate models available through AWS Bedrock (Claude, Llama, DeepSeek, Titan, ...):

```bash
# Evaluate all supported Bedrock models
python bedrock_privacy_evaluation.py --json_path questionnaire.json --output_dir ./results

# Evaluate specific model
python bedrock_privacy_evaluation.py \
    --json_path questionnaire.json \
    --single_model us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
    --output_dir ./results

# Custom sampling and timeout
python bedrock_privacy_evaluation.py \
    --json_path questionnaire.json \
    --num_samples 5 \
    --default_timeout 120 \
    --output_dir ./results
```

### Local Model Evaluation
Evaluate local models (BERT, RoBERTa, GPT-2, etc.):

```bash
# Evaluate all local model types
python local_model_privacy_evaluation.py --json_path questionnaire.json --output_dir ./results

# Evaluate only encoder models (BERT, RoBERTa)
python local_model_privacy_evaluation.py \
    --json_path questionnaire.json \
    --skip_gpt2 --skip_generative --skip_openai \
    --output_dir ./results

# Custom device and sampling
python local_model_privacy_evaluation.py \
    --json_path questionnaire.json \
    --device cuda \
    --num_samples 5 \
    --output_dir ./results
```

### Generate Visualizations
Create charts and plots from evaluation results:

```bash
# Generate all visualizations
python visualize_privacy_scores.py

# The script will automatically look for results in ./results/privacy_attitudes_results.json
# and save visualizations to ./results/visualizations/
```

## 🎯 Supported Models

### AWS Bedrock
- DeepSeek R1 (`us.deepseek.r1-v1:0`)
- Claude 3.7 Sonnet (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
- Llama 3.1 405B (`meta.llama3-1-405b-instruct-v1:0`)
- Amazon Titan Text Express (`amazon.titan-text-express-v1`)
- For more model ref https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

### Local/OpenAI Models
- **Encoder**: BERT, RoBERTa, DistilBERT, ALBERT
- **GPT-2**: GPT-2, GPT-2 Medium
- **Generative**: Mistral 7B, Llama 3.1 8B (requires local installation)
- **OpenAI API**: GPT-4, GPT-3.5, GPT-4o


## 🔧 Command Line Options

### Common Options
- `--json_path`: Path to questionnaire JSON file (required)
- `--output_dir`: Output directory for results
- `--num_samples`: Number of response samples per question (default: 10)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Bedrock-specific Options
- `--single_model`: Evaluate specific model only
- `--region_name`: AWS region (default: us-west-2)
- `--default_timeout`: API timeout in seconds
- `--model_timeouts`: Custom timeouts per model (JSON format)

### Local Model Options
- `--skip_encoder`: Skip encoder model evaluation
- `--skip_gpt2`: Skip GPT-2 model evaluation
- `--skip_generative`: Skip local generative model evaluation
- `--skip_openai`: Skip OpenAI API model evaluation
- `--device`: Computation device (auto, cuda, cpu)
- `--temperature`: Text generation temperature (default: 0.7)

## 🛠️ Troubleshooting

### Common Issues

**AWS Credentials Error:**
```bash
# Verify AWS credentials
aws sts get-caller-identity
```

**CUDA Memory Error:**
```bash
# Use CPU instead
python local_model_privacy_evaluation.py --json_path questionnaire.json --device cpu
```

**OpenAI API Rate Limit:**
```bash
# Reduce sampling rate
python local_model_privacy_evaluation.py --json_path questionnaire.json --num_samples 3
```

### Debug Mode
```bash
# Enable debug logging
python bedrock_privacy_evaluation.py --json_path questionnaire.json --log_level DEBUG
```

## 📚 References

This work is based on the Internet Users' Information Privacy Concerns (IUIPC) framework:

> Malhotra, N. K., Kim, S. S., & Agarwal, J. (2004). Internet users' information privacy concerns (IUIPC): The construct, the scale, and a causal model. *Information Systems Research*, 15(4), 336-355. https://pubsonline.informs.org/doi/abs/10.1287/isre.1040.0032
