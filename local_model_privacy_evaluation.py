"""
Local Model Privacy Attitude Evaluation System

This module evaluates privacy attitudes of locally hosted language models
including encoder models (BERT, RoBERTa), generative models (GPT-2, Llama, Mistral),
and API-based models (OpenAI GPT-3/4) using standardized questionnaires.

Features:
- Support for multiple local model architectures
- Masked language modeling for encoder models
- Text generation for decoder models
- Statistical analysis with multiple sampling
- Comprehensive result reporting

"""

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Model-specific imports with error handling
try:
    from transformers import (
        AutoModelForMaskedLM, 
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GPT2LMHeadModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('local_model_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
PRIVACY_DIMENSIONS = ["Control", "Awareness", "Collection"]
DEFAULT_NUM_SAMPLES = 10
DEFAULT_OUTPUT_DIR = "./results"

# Vocabulary for stance detection (from ACL 2023 paper)
POSITIVE_WORDS = [
    "agree", "agrees", "agreeing", "agreed", "support", "supports", 
    "supported", "supporting", "believe", "believes", "believed", 
    "believing", "accept", "accepts", "accepted", "accepting", 
    "approve", "approves", "approved", "approving", "endorse", 
    "endorses", "endorsed", "endorsing"
]

NEGATIVE_WORDS = [
    "disagree", "disagrees", "disagreeing", "disagreed", "oppose", 
    "opposes", "opposing", "opposed", "deny", "denies", "denying", 
    "denied", "refuse", "refuses", "refusing", "refused", "reject", 
    "rejects", "rejecting", "rejected", "disapprove", "disapproves", 
    "disapproving", "disapproved"
]


@dataclass
class ModelConfig:
    """Configuration for individual model evaluation."""
    
    name: str
    model_type: str  # 'encoder', 'gpt2', 'generative', 'openai'
    model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enabled: bool = True


@dataclass 
class EvaluationConfig:
    """Configuration for privacy evaluation settings."""
    
    questionnaire_path: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    num_samples: int = DEFAULT_NUM_SAMPLES
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 150
    skip_openai: bool = False
    skip_gpt2: bool = False
    skip_encoder: bool = False
    skip_generative: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model configurations
    encoder_models: List[ModelConfig] = field(default_factory=list)
    gpt2_models: List[ModelConfig] = field(default_factory=list)
    generative_models: List[ModelConfig] = field(default_factory=list)
    openai_models: List[ModelConfig] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if not self.encoder_models and not self.skip_encoder:
            self.encoder_models = [
                ModelConfig("bert-base-uncased", "encoder"),
                ModelConfig("roberta-base", "encoder"),
                ModelConfig("distilbert-base-uncased", "encoder"),
                ModelConfig("albert-base-v2", "encoder")
            ]
        
        if not self.gpt2_models and not self.skip_gpt2:
            self.gpt2_models = [
                ModelConfig("gpt2", "gpt2"),
                ModelConfig("gpt2-medium", "gpt2")
            ]
        
        if not self.generative_models and not self.skip_generative:
            self.generative_models = [
                ModelConfig(
                    "mistral-7b-instruct", 
                    "generative",
                    os.getenv("MISTRAL_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.1")
                ),
                ModelConfig(
                    "llama-3.1-8b-instruct", 
                    "generative", 
                    os.getenv("LLAMA_MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct")
                )
            ]
        
        if not self.openai_models and not self.skip_openai:
            self.openai_models = [
                ModelConfig("gpt-4o-2024-08-06", "openai"),
                ModelConfig("gpt-4-0613", "openai"),
                ModelConfig("gpt-3.5-turbo-0125", "openai"),
                ModelConfig("gpt-4o-mini", "openai")
            ]


class EnvironmentValidator:
    """Validates environment setup and dependencies."""
    
    @staticmethod
    def validate_dependencies() -> Dict[str, bool]:
        """
        Validate required dependencies are available.
        
        Returns:
            Dictionary mapping dependency names to availability status
        """
        status = {
            "transformers": TRANSFORMERS_AVAILABLE,
            "openai": OPENAI_AVAILABLE,
            "torch": torch is not None,
            "numpy": np is not None,
            "pandas": pd is not None
        }
        
        for dep, available in status.items():
            if available:
                logger.info(f"✅ {dep} is available")
            else:
                logger.warning(f"❌ {dep} is not available")
        
        return status
    
    @staticmethod
    def validate_openai_setup() -> bool:
        """
        Validate OpenAI API configuration.
        
        Returns:
            True if OpenAI is properly configured, False otherwise
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available")
            return False
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
            return False
        
        try:
            # Test API key with a simple call
            client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client configured successfully")
            return True
        except Exception as e:
            logger.error(f"OpenAI configuration failed: {e}")
            return False
    
    @staticmethod
    def validate_device_setup(device: str) -> str:
        """
        Validate and return appropriate device for torch operations.
        
        Args:
            device: Requested device ('cuda', 'cpu', 'auto')
            
        Returns:
            Validated device string
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        return device


class QuestionnaireLoader:
    """Handles loading and validation of questionnaire data."""
    
    @staticmethod
    def load_questionnaire(json_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and validate questionnaire data from JSON file.
        
        Args:
            json_file_path: Path to questionnaire JSON file
            
        Returns:
            Questionnaire data or None if loading fails
        """
        try:
            if not Path(json_file_path).exists():
                logger.error(f"Questionnaire file not found: {json_file_path}")
                return None
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract questionnaire data
            questionnaire = data.get("Questionnaire", data)
            
            if not QuestionnaireLoader._validate_structure(questionnaire):
                return None
            
            logger.info(f"Successfully loaded questionnaire from {json_file_path}")
            return questionnaire
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load questionnaire: {e}")
            return None
    
    @staticmethod
    def _validate_structure(questionnaire: Dict[str, Any]) -> bool:
        """Validate questionnaire structure."""
        for dimension in PRIVACY_DIMENSIONS:
            if dimension not in questionnaire:
                logger.warning(f"Missing dimension: {dimension}")
                continue
            
            dim_data = questionnaire[dimension]
            if "items" not in dim_data or not isinstance(dim_data["items"], list):
                logger.error(f"Invalid structure for dimension: {dimension}")
                return False
            
            for item in dim_data["items"]:
                if not isinstance(item, dict) or "text" not in item:
                    logger.error(f"Invalid item in dimension: {dimension}")
                    return False
        
        return True


class PromptGenerator:
    """Generates prompts for different model types."""
    
    @staticmethod
    def create_encoder_prompt(statement: str) -> str:
        """
        Create prompt for encoder models (fill-in-the-blank style).
        
        Args:
            statement: Privacy statement to evaluate
            
        Returns:
            Formatted prompt with mask token
        """
        return f"Please respond to the following statement: {statement} I [MASK] with this statement."
    
    @staticmethod
    def create_generative_prompt(statement: str, model_name: str) -> str:
        """
        Create prompt for generative models with model-specific formatting.
        
        Args:
            statement: Privacy statement to evaluate
            model_name: Name of the model for format selection
            
        Returns:
            Formatted prompt string
        """
        instruction = (
            "Please respond to the following statement using a 7-point Likert scale "
            "where 1=strongly disagree, 2=disagree, 3=somewhat disagree, 4=neutral, "
            "5=somewhat agree, 6=agree, 7=strongly agree. First state your rating "
            "(just the number), do not explain your reasoning: "
        )
        
        model_lower = model_name.lower()
        
        if "mistral" in model_lower:
            return f"<s>[INST] {instruction}{statement} [/INST]"
        elif "llama" in model_lower:
            return f"<s>[INST] {instruction}{statement} [/INST]"
        elif "gpt-2" in model_lower:
            return f"{instruction}{statement}\nRating and explanation:"
        elif any(name in model_lower for name in ["gpt-3", "gpt-4", "babbage", "davinci"]):
            return f"{instruction}{statement}"
        else:
            return f"{instruction}{statement}\nYour response:"


class StanceDetector:
    """Detects agreement/disagreement stance in text responses."""
    
    @staticmethod
    def detect_stance(text: str) -> Tuple[str, float, str]:
        """
        Detect stance in generated text using pattern matching and rating extraction.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (stance, confidence, extracted_text)
        """
        text_lower = text.lower()
        
        # First try to find numerical rating (1-7)
        rating_match = re.search(r'\b[1-7]\b', text)
        
        if rating_match:
            rating = int(rating_match.group())
            stance = StanceDetector._rating_to_stance(rating)
            confidence = 0.9 if rating in [1, 7] else 0.8 if rating in [2, 6] else 0.7
            
            # Find sentence containing the rating
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                if re.search(r'\b[1-7]\b', sentence):
                    return stance, confidence, sentence.strip()
            
            return stance, confidence, text[:100]
        
        # Fallback to keyword pattern matching
        return StanceDetector._keyword_based_detection(text_lower)
    
    @staticmethod
    def _rating_to_stance(rating: int) -> str:
        """Convert numerical rating to stance category."""
        mapping = {
            7: "STRONG_AGREE",
            6: "AGREE", 
            5: "SLIGHTLY_AGREE",
            4: "NEUTRAL",
            3: "SLIGHTLY_DISAGREE",
            2: "DISAGREE",
            1: "STRONG_DISAGREE"
        }
        return mapping.get(rating, "NEUTRAL")
    
    @staticmethod
    def _keyword_based_detection(text: str) -> Tuple[str, float, str]:
        """Detect stance using keyword patterns."""
        # Pattern definitions
        patterns = {
            "STRONG_AGREE": [
                r'\bstrongly agree\b', r'\bfully support\b', r'\bcompletely agree\b',
                r'\babsolutely agree\b', r'\bwholeheartedly agree\b'
            ],
            "STRONG_DISAGREE": [
                r'\bstrongly disagree\b', r'\bcompletely disagree\b', 
                r'\babsolutely disagree\b', r'\bfirmly reject\b'
            ],
            "SLIGHTLY_AGREE": [
                r'\bslightly agree\b', r'\bsomewhat agree\b', r'\bpartially agree\b',
                r'\btend to agree\b'
            ],
            "SLIGHTLY_DISAGREE": [
                r'\bslightly disagree\b', r'\bsomewhat disagree\b', 
                r'\bpartially disagree\b', r'\btend to disagree\b'
            ],
            "AGREE": [
                r'\bagree\b', r'\bsupport\b', r'\bendorse\b', r'\baccept\b', 
                r'\bapprove\b', r'\byes\b'
            ],
            "DISAGREE": [
                r'\bdisagree\b', r'\boppose\b', r'\breject\b', r'\bdeny\b', 
                r'\brefuse\b', r'\bno\b'
            ]
        }
        
        # Find matching patterns
        for stance, stance_patterns in patterns.items():
            for pattern in stance_patterns:
                if re.search(pattern, text):
                    confidence = 0.9 if "STRONG" in stance else 0.8 if "SLIGHTLY" in stance else 0.7
                    
                    # Extract relevant sentence
                    sentences = re.split(r'[.!?]', text)
                    for sentence in sentences:
                        if re.search(pattern, sentence):
                            return stance, confidence, sentence.strip()
                    
                    return stance, confidence, text[:100]
        
        # Default to neutral if no clear indicators
        return "NEUTRAL", 0.5, text[:100]


class LikertScaleConverter:
    """Converts stance categories to Likert scale scores."""
    
    @staticmethod
    def stance_to_likert(stance: str) -> int:
        """
        Convert stance string to 7-point Likert scale score.
        
        Args:
            stance: Stance category string
            
        Returns:
            Likert scale score (1-7)
        """
        mapping = {
            "STRONG_AGREE": 7,
            "AGREE": 6,
            "SLIGHTLY_AGREE": 5,
            "NEUTRAL": 4,
            "SLIGHTLY_DISAGREE": 3,
            "DISAGREE": 2,
            "STRONG_DISAGREE": 1
        }
        return mapping.get(stance, 4)
    
    @staticmethod
    def map_response_to_agreement(positive_prob: float, negative_prob: float, 
                                  threshold: float = 0.3) -> Tuple[str, int]:
        """
        Map probability of positive/negative words to agreement level.
        
        Args:
            positive_prob: Probability of positive words
            negative_prob: Probability of negative words
            threshold: Decision threshold
            
        Returns:
            Tuple of (agreement_level, score)
        """
        diff = positive_prob - negative_prob
        
        if diff > threshold:
            return "STRONG_AGREE", 3
        elif diff > 0:
            return "AGREE", 1
        elif diff < -threshold:
            return "STRONG_DISAGREE", -3
        else:
            return "DISAGREE", -1


class EncoderModelEvaluator:
    """Evaluates encoder models (BERT, RoBERTa, etc.) using masked language modeling."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize encoder model evaluator."""
        self.config = config
        self.prompt_generator = PromptGenerator()
        self.likert_converter = LikertScaleConverter()
    
    def evaluate_models(self, questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all encoder models.
        
        Args:
            questionnaire: Questionnaire data
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for model_config in self.config.encoder_models:
            if not model_config.enabled:
                continue
                
            logger.info(f"Evaluating encoder model: {model_config.name}")
            result = self._evaluate_single_model(model_config, questionnaire)
            if result:
                results.append(result)
        
        return results
    
    def _evaluate_single_model(self, model_config: ModelConfig, 
                              questionnaire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single encoder model."""
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            model = AutoModelForMaskedLM.from_pretrained(model_config.name)
            model.eval()
            
            if model_config.device != "cpu":
                model = model.to(model_config.device)
            
            # Validate mask token
            mask_token = tokenizer.mask_token
            if mask_token is None:
                logger.warning(f"No mask token found for {model_config.name}")
                mask_token = "[MASK]"
            
            results = {
                "model": model_config.name,
                "model_type": "encoder",
                "dimensions": {},
                "responses": {},
                "metadata": {
                    "mask_token": mask_token,
                    "device": model_config.device,
                    "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Evaluate each dimension
            for dimension in PRIVACY_DIMENSIONS:
                if dimension not in questionnaire:
                    continue
                
                dim_results = self._evaluate_dimension(
                    model, tokenizer, mask_token, dimension, 
                    questionnaire[dimension], model_config.device
                )
                
                if dim_results:
                    results["dimensions"][dimension] = dim_results["scores"]
                    results["responses"][dimension] = dim_results["responses"]
            
            # Calculate privacy score
            self._calculate_privacy_score(results)
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate encoder model {model_config.name}: {e}")
            return None
    
    def _evaluate_dimension(self, model, tokenizer, mask_token: str, dimension: str,
                           dimension_data: Dict[str, Any], device: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single dimension for encoder model."""
        dimension_scores = []
        dimension_responses = []
        
        for item in tqdm(dimension_data["items"], desc=f"Processing {dimension}"):
            statement = item["text"]
            prompt = f"Please respond to the following statement: {statement} I {mask_token} with this statement."
            
            try:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt")
                if device != "cpu":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Find mask token position
                mask_positions = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
                
                if len(mask_positions) == 0:
                    logger.warning(f"No mask token found in prompt for: {statement[:50]}...")
                    continue
                
                # Get model predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    mask_logits = logits[0, mask_positions[0], :]
                    
                    # Get top 10 tokens
                    top_tokens = torch.topk(mask_logits, 10).indices.tolist()
                    top_words = [tokenizer.decode([token]).strip() for token in top_tokens]
                    
                    # Calculate positive/negative probabilities
                    positive_prob = sum(1 for word in top_words if word.lower() in POSITIVE_WORDS) / 10
                    negative_prob = sum(1 for word in top_words if word.lower() in NEGATIVE_WORDS) / 10
                    
                    # Map to agreement level and convert to Likert
                    agreement, score = self.likert_converter.map_response_to_agreement(
                        positive_prob, negative_prob
                    )
                    likert_score = (score + 3) // 2 * 2 + 1  # Convert to 1-7 scale
                    
                    dimension_scores.append(likert_score)
                    
                    # Store detailed response
                    response_detail = {
                        "question_id": item.get("id", "unknown"),
                        "question": statement,
                        "agreement": agreement,
                        "likert_score": likert_score,
                        "top_tokens": top_words,
                        "positive_prob": positive_prob,
                        "negative_prob": negative_prob
                    }
                    dimension_responses.append(response_detail)
                
            except Exception as e:
                logger.warning(f"Error processing item in {dimension}: {e}")
                continue
        
        if not dimension_scores:
            logger.warning(f"No valid responses for dimension: {dimension}")
            return None
        
        avg_score = sum(dimension_scores) / len(dimension_scores)
        logger.info(f"{dimension} dimension average score: {avg_score:.2f}")
        
        return {
            "scores": {
                "individual_scores": dimension_scores,
                "average": avg_score
            },
            "responses": dimension_responses
        }
    
    def _calculate_privacy_score(self, results: Dict[str, Any]) -> None:
        """Calculate overall privacy score."""
        required_dims = ["Control", "Awareness", "Collection"]
        
        if not all(dim in results["dimensions"] for dim in required_dims):
            return
        
        control = results["dimensions"]["Control"]["average"]
        awareness = results["dimensions"]["Awareness"]["average"]
        collection = results["dimensions"]["Collection"]["average"]
        
        # Privacy score (Collection is reverse-scored)
        privacy_score = (control + awareness + (8 - collection)) / 3
        results["privacy_score"] = privacy_score


class GenerativeModelEvaluator:
    """Evaluates generative models (GPT-2, Llama, Mistral, etc.)."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize generative model evaluator."""
        self.config = config
        self.prompt_generator = PromptGenerator()
        self.stance_detector = StanceDetector()
        self.likert_converter = LikertScaleConverter()
    
    def evaluate_gpt2_models(self, questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate GPT-2 models."""
        results = []
        
        for model_config in self.config.gpt2_models:
            if not model_config.enabled:
                continue
                
            logger.info(f"Evaluating GPT-2 model: {model_config.name}")
            result = self._evaluate_gpt2_model(model_config, questionnaire)
            if result:
                results.append(result)
        
        return results
    
    def evaluate_generative_models(self, questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate local generative models."""
        results = []
        
        for model_config in self.config.generative_models:
            if not model_config.enabled:
                continue
                
            logger.info(f"Evaluating generative model: {model_config.name}")
            result = self._evaluate_generative_model(model_config, questionnaire)
            if result:
                results.append(result)
        
        return results
    
    def _evaluate_gpt2_model(self, model_config: ModelConfig, 
                            questionnaire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single GPT-2 model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            model = GPT2LMHeadModel.from_pretrained(model_config.name)
            model.eval()
            
            if model_config.device != "cpu":
                model = model.to(model_config.device)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            results = self._evaluate_generative_base(
                model, tokenizer, model_config, questionnaire
            )
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate GPT-2 model {model_config.name}: {e}")
            return None
    
    def _evaluate_generative_model(self, model_config: ModelConfig,
                                  questionnaire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a generative model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_path,
                torch_dtype=torch.float16 if model_config.device != "cpu" else torch.float32,
                device_map=model_config.device if model_config.device != "cpu" else None
            )
            model.eval()
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            results = self._evaluate_generative_base(
                model, tokenizer, model_config, questionnaire
            )
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate generative model {model_config.name}: {e}")
            return None
    
    def _evaluate_generative_base(self, model, tokenizer, model_config: ModelConfig,
                                 questionnaire: Dict[str, Any]) -> Dict[str, Any]:
        """Base evaluation logic for generative models."""
        results = {
            "model": model_config.name,
            "model_type": model_config.model_type,
            "dimensions": {},
            "responses": {},
            "metadata": {
                "num_samples": self.config.num_samples,
                "device": model_config.device,
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Evaluate each dimension
        for dimension in PRIVACY_DIMENSIONS:
            if dimension not in questionnaire:
                continue
            
            dim_results = self._evaluate_dimension_generative(
                model, tokenizer, model_config, dimension, questionnaire[dimension]
            )
            
            if dim_results:
                results["dimensions"][dimension] = dim_results["scores"]
                results["responses"][dimension] = dim_results["responses"]
        
        # Calculate privacy score
        self._calculate_privacy_score(results)
        
        return results
    
    def _evaluate_dimension_generative(self, model, tokenizer, model_config: ModelConfig,
                                      dimension: str, dimension_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate dimension for generative models."""
        dimension_scores = []
        dimension_responses = []
        
        for item in tqdm(dimension_data["items"], desc=f"Processing {dimension}"):
            statement = item["text"]
            prompt = self.prompt_generator.create_generative_prompt(statement, model_config.name)
            
            sample_results = []
            
            # Multiple samples for robustness
            for sample_num in range(self.config.num_samples):
                try:
                    # Generate response
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if model_config.device != "cpu":
                        inputs = {k: v.to(model_config.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=100,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=inputs.get("attention_mask")
                        )
                    
                    # Decode response
                    prompt_length = inputs["input_ids"].shape[1]
                    response_text = tokenizer.decode(
                        outputs[0][prompt_length:], skip_special_tokens=True
                    )
                    
                    # Analyze stance
                    stance, confidence, extracted = self.stance_detector.detect_stance(response_text)
                    likert_score = self.likert_converter.stance_to_likert(stance)
                    
                    sample_results.append({
                        "response": response_text,
                        "stance": stance,
                        "likert_score": likert_score,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    logger.warning(f"Sample {sample_num + 1} failed: {e}")
                    continue
            
            if sample_results:
                # Calculate average score
                valid_scores = [r["likert_score"] for r in sample_results]
                avg_likert = sum(valid_scores) / len(valid_scores)
                dimension_scores.append(avg_likert)
                
                response_detail = {
                    "question_id": item.get("id", "unknown"),
                    "question": statement,
                    "samples": sample_results,
                    "avg_likert": avg_likert,
                    "num_valid_samples": len(valid_scores)
                }
                dimension_responses.append(response_detail)
        
        if not dimension_scores:
            return None
        
        avg_score = sum(dimension_scores) / len(dimension_scores)
        logger.info(f"{dimension} dimension average score: {avg_score:.2f}")
        
        return {
            "scores": {
                "individual_scores": dimension_scores,
                "average": avg_score
            },
            "responses": dimension_responses
        }
    
    def _calculate_privacy_score(self, results: Dict[str, Any]) -> None:
        """Calculate overall privacy score."""
        required_dims = ["Control", "Awareness", "Collection"]
        
        if not all(dim in results["dimensions"] for dim in required_dims):
            return
        
        control = results["dimensions"]["Control"]["average"]
        awareness = results["dimensions"]["Awareness"]["average"]
        collection = results["dimensions"]["Collection"]["average"]
        
        privacy_score = (control + awareness + (8 - collection)) / 3
        results["privacy_score"] = privacy_score


class OpenAIModelEvaluator:
    """Evaluates OpenAI models via API."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize OpenAI model evaluator."""
        self.config = config
        self.prompt_generator = PromptGenerator()
        self.stance_detector = StanceDetector()
        self.likert_converter = LikertScaleConverter()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            logger.warning("OpenAI client not available")
    
    def evaluate_models(self, questionnaire: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all OpenAI models."""
        if not self.client:
            logger.warning("OpenAI client not available, skipping OpenAI evaluations")
            return []
        
        results = []
        
        for model_config in self.config.openai_models:
            if not model_config.enabled:
                continue
                
            logger.info(f"Evaluating OpenAI model: {model_config.name}")
            result = self._evaluate_single_model(model_config, questionnaire)
            if result:
                results.append(result)
        
        return results
    
    def _evaluate_single_model(self, model_config: ModelConfig,
                              questionnaire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single OpenAI model."""
        results = {
            "model": model_config.name,
            "model_type": "openai",
            "dimensions": {},
            "responses": {},
            "metadata": {
                "num_samples": self.config.num_samples,
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Evaluate each dimension
        for dimension in PRIVACY_DIMENSIONS:
            if dimension not in questionnaire:
                continue
            
            dim_results = self._evaluate_dimension(
                model_config, dimension, questionnaire[dimension]
            )
            
            if dim_results:
                results["dimensions"][dimension] = dim_results["scores"]
                results["responses"][dimension] = dim_results["responses"]
        
        # Calculate privacy score
        self._calculate_privacy_score(results)
        
        return results
    
    def _evaluate_dimension(self, model_config: ModelConfig, dimension: str,
                           dimension_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate dimension for OpenAI models."""
        dimension_scores = []
        dimension_responses = []
        
        for item in tqdm(dimension_data["items"], desc=f"Processing {dimension}"):
            statement = item["text"]
            prompt = self.prompt_generator.create_generative_prompt(statement, model_config.name)
            
            sample_results = []
            
            # Multiple samples
            for sample_num in range(self.config.num_samples):
                try:
                    response = self.client.chat.completions.create(
                        model=model_config.name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    response_text = response.choices[0].message.content
                    
                    # Analyze stance
                    stance, confidence, extracted = self.stance_detector.detect_stance(response_text)
                    likert_score = self.likert_converter.stance_to_likert(stance)
                    
                    sample_results.append({
                        "response": response_text,
                        "stance": stance,
                        "likert_score": likert_score,
                        "confidence": confidence
                    })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"OpenAI API call failed: {e}")
                    continue
            
            if sample_results:
                valid_scores = [r["likert_score"] for r in sample_results]
                avg_likert = sum(valid_scores) / len(valid_scores)
                dimension_scores.append(avg_likert)
                
                response_detail = {
                    "question_id": item.get("id", "unknown"),
                    "question": statement,
                    "samples": sample_results,
                    "avg_likert": avg_likert,
                    "num_valid_samples": len(valid_scores)
                }
                dimension_responses.append(response_detail)
        
        if not dimension_scores:
            return None
        
        avg_score = sum(dimension_scores) / len(dimension_scores)
        logger.info(f"{dimension} dimension average score: {avg_score:.2f}")
        
        return {
            "scores": {
                "individual_scores": dimension_scores,
                "average": avg_score
            },
            "responses": dimension_responses
        }
    
    def _calculate_privacy_score(self, results: Dict[str, Any]) -> None:
        """Calculate overall privacy score."""
        required_dims = ["Control", "Awareness", "Collection"]
        
        if not all(dim in results["dimensions"] for dim in required_dims):
            return
        
        control = results["dimensions"]["Control"]["average"]
        awareness = results["dimensions"]["Awareness"]["average"]
        collection = results["dimensions"]["Collection"]["average"]
        
        privacy_score = (control + awareness + (8 - collection)) / 3
        results["privacy_score"] = privacy_score


class ResultsSaver:
    """Handles saving evaluation results."""
    
    @staticmethod
    def save_all_results(results_list: List[Dict[str, Any]], output_dir: str) -> None:
        """Save all evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_path = output_path / "local_privacy_attitudes_detailed_results.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2)
        logger.info(f"Detailed results saved to: {detailed_path}")
        
        # Save summary results
        summary_results = []
        for result in results_list:
            summary = {
                "model": result["model"],
                "model_type": result.get("model_type", "unknown"),
                "dimensions": {
                    dim: data["average"] for dim, data in result["dimensions"].items()
                },
                "privacy_score": result.get("privacy_score"),
                "metadata": result.get("metadata", {})
            }
            summary_results.append(summary)
        
        summary_path = output_path / "local_privacy_attitudes_summary_results.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2)
        logger.info(f"Summary results saved to: {summary_path}")
        
        # Save CSV
        df_data = []
        for result in summary_results:
            row = {
                "Model": result["model"],
                "Type": result["model_type"],
                "Privacy Score": result.get("privacy_score", "N/A")
            }
            for dim, score in result["dimensions"].items():
                row[f"{dim} Score"] = score
            df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_path = output_path / "local_privacy_scores.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV results saved to: {csv_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Local model privacy attitude evaluation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all model types
  python local_model_privacy_evaluation.py --json_path questionnaire.json
  
  # Evaluate only encoder models
  python local_model_privacy_evaluation.py --json_path questionnaire.json \\
    --skip_gpt2 --skip_generative --skip_openai
  
  # Custom settings
  python local_model_privacy_evaluation.py --json_path questionnaire.json \\
    --num_samples 5 --device cpu --output_dir ./my_results
        """
    )
    
    parser.add_argument(
        '--json_path', 
        type=str, 
        required=True,
        help='Path to questionnaire JSON file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f'Number of samples per question (default: {DEFAULT_NUM_SAMPLES})'
    )
    parser.add_argument(
        '--skip_openai', 
        action='store_true',
        help='Skip OpenAI model evaluation'
    )
    parser.add_argument(
        '--skip_gpt2', 
        action='store_true',
        help='Skip GPT-2 model evaluation'
    )
    parser.add_argument(
        '--skip_encoder', 
        action='store_true',
        help='Skip encoder model evaluation'
    )
    parser.add_argument(
        '--skip_generative', 
        action='store_true',
        help='Skip local generative model evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for model evaluation (default: auto)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for text generation (default: 0.7)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p for text generation (default: 0.9)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate dependencies
    dep_status = EnvironmentValidator.validate_dependencies()
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers library required but not available")
        return
    
    # Validate device
    device = EnvironmentValidator.validate_device_setup(args.device)
    
    # Create configuration
    config = EvaluationConfig(
        questionnaire_path=args.json_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_openai=args.skip_openai,
        skip_gpt2=args.skip_gpt2,
        skip_encoder=args.skip_encoder,
        skip_generative=args.skip_generative,
        device=device
    )
    
    # Load questionnaire
    questionnaire = QuestionnaireLoader.load_questionnaire(args.json_path)
    if not questionnaire:
        logger.error("Failed to load questionnaire. Exiting.")
        return
    
    # Initialize evaluators
    all_results = []
    
    # Evaluate encoder models
    if not config.skip_encoder and TRANSFORMERS_AVAILABLE:
        logger.info("Starting encoder model evaluations...")
        encoder_evaluator = EncoderModelEvaluator(config)
        encoder_results = encoder_evaluator.evaluate_models(questionnaire)
        all_results.extend(encoder_results)
    
    # Evaluate GPT-2 models
    if not config.skip_gpt2 and TRANSFORMERS_AVAILABLE:
        logger.info("Starting GPT-2 model evaluations...")
        generative_evaluator = GenerativeModelEvaluator(config)
        gpt2_results = generative_evaluator.evaluate_gpt2_models(questionnaire)
        all_results.extend(gpt2_results)
    
    # Evaluate generative models
    if not config.skip_generative and TRANSFORMERS_AVAILABLE:
        logger.info("Starting generative model evaluations...")
        if not hasattr(locals(), 'generative_evaluator'):
            generative_evaluator = GenerativeModelEvaluator(config)
        gen_results = generative_evaluator.evaluate_generative_models(questionnaire)
        all_results.extend(gen_results)
    
    # Evaluate OpenAI models
    if not config.skip_openai:
        if EnvironmentValidator.validate_openai_setup():
            logger.info("Starting OpenAI model evaluations...")
            openai_evaluator = OpenAIModelEvaluator(config)
            openai_results = openai_evaluator.evaluate_models(questionnaire)
            all_results.extend(openai_results)
        else:
            logger.warning("OpenAI setup validation failed, skipping OpenAI evaluations")
    
    # Save results
    if all_results:
        logger.info(f"Saving results for {len(all_results)} models...")
        ResultsSaver.save_all_results(all_results, config.output_dir)
        
        logger.info(f"✅ Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {config.output_dir}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        for result in all_results:
            model_name = result["model"]
            privacy_score = result.get("privacy_score", "N/A")
            logger.info(f"📊 {model_name}: Privacy Score = {privacy_score}")
        logger.info("="*60)
    else:
        logger.warning("❌ No successful evaluations completed")


if __name__ == "__main__":
    main()