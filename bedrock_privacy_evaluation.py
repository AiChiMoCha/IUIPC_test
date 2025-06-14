"""
Privacy Attitude Evaluation for Large Language Models using AWS Bedrock API

This module evaluates the privacy attitudes of language models available through
AWS Bedrock by administering standardized privacy questionnaires and analyzing
responses using a 7-point Likert scale.

"""

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import boto3
import dspy
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bedrock_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REGION = "us-west-2"
DEFAULT_TIMEOUT = 60
DEFAULT_SAMPLES = 10
PRIVACY_DIMENSIONS = ["Control", "Awareness", "Collection"]

# Model timeout configurations (in seconds)
DEFAULT_MODEL_TIMEOUTS = {
    "anthropic": 3600,  # 60 minutes for Claude models
    "meta": 300,        # 5 minutes for Llama models  
    "deepseek": 300,    # 5 minutes for DeepSeek models
    "amazon": 120       # 2 minutes for Titan models
}

# Supported models list
SUPPORTED_MODELS = [
    "us.deepseek.r1-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
    "meta.llama3-1-405b-instruct-v1:0",
    "amazon.titan-text-express-v1"
    "Extension ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"
]


@dataclass
class EvaluationConfig:
    """Configuration class for privacy evaluation settings."""
    
    region_name: str = DEFAULT_REGION
    num_samples: int = DEFAULT_SAMPLES
    default_timeout: int = DEFAULT_TIMEOUT
    model_timeouts: Dict[str, int] = None
    output_dir: str = "./results"
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.model_timeouts is None:
            self.model_timeouts = {}


class EnvironmentValidator:
    """Validates required environment variables and configurations."""
    
    @staticmethod
    def validate_aws_credentials() -> bool:
        """
        Validate AWS credentials are properly configured.
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing AWS credentials: {', '.join(missing_vars)}")
            return False
        
        try:
            # Test AWS credentials by creating a client
            boto3.client('sts').get_caller_identity()
            logger.info("AWS credentials validated successfully")
            return True
        except Exception as e:
            logger.error(f"AWS credential validation failed: {e}")
            return False
    
    @staticmethod
    def validate_dspy_setup() -> bool:
        """
        Validate DSPy setup for stance classification.
        
        Returns:
            bool: True if DSPy is properly configured, False otherwise
        """
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                logger.error("OPENAI_API_KEY not found for DSPy configuration")
                return False
            
            # Initialize DSPy with GPT-4o-mini
            lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
            dspy.configure(lm=lm)
            logger.info("DSPy configured successfully")
            return True
        except Exception as e:
            logger.error(f"DSPy configuration failed: {e}")
            return False


class LikertStance(dspy.Signature):
    """DSPy signature for classifying responses on a 7-point Likert scale."""
    
    response: str = dspy.InputField(desc="The response text to classify")
    likert_rating: str = dspy.OutputField(
        desc="Rating from 1-7 where 1=strongly disagree, 7=strongly agree"
    )
    stance: str = dspy.OutputField(
        desc="Categorical stance: STRONG_DISAGREE, DISAGREE, SLIGHTLY_DISAGREE, "
             "NEUTRAL, SLIGHTLY_AGREE, AGREE, STRONG_AGREE"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )


class QuestionnaireLoader:
    """Handles loading and validation of questionnaire data."""
    
    @staticmethod
    def load_questionnaire(json_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load questionnaire data from JSON file.
        
        Args:
            json_file_path: Path to the questionnaire JSON file
            
        Returns:
            Dictionary containing questionnaire data or None if loading fails
        """
        try:
            if not os.path.exists(json_file_path):
                logger.error(f"Questionnaire file not found: {json_file_path}")
                return None
                
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate structure
            if "Questionnaire" in data:
                questionnaire = data["Questionnaire"]
            else:
                questionnaire = data
                
            if not QuestionnaireLoader._validate_questionnaire(questionnaire):
                return None
                
            logger.info(f"Successfully loaded questionnaire from {json_file_path}")
            return questionnaire
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in questionnaire file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load questionnaire: {e}")
            return None
    
    @staticmethod
    def _validate_questionnaire(questionnaire: Dict[str, Any]) -> bool:
        """
        Validate questionnaire structure.
        
        Args:
            questionnaire: Questionnaire data to validate
            
        Returns:
            True if valid, False otherwise
        """
        for dimension in PRIVACY_DIMENSIONS:
            if dimension not in questionnaire:
                logger.warning(f"Missing dimension in questionnaire: {dimension}")
                continue
                
            if "items" not in questionnaire[dimension]:
                logger.error(f"Missing 'items' in dimension: {dimension}")
                return False
                
            items = questionnaire[dimension]["items"]
            if not isinstance(items, list) or len(items) == 0:
                logger.error(f"Empty or invalid items in dimension: {dimension}")
                return False
                
            # Validate item structure
            for item in items:
                if not isinstance(item, dict) or "text" not in item:
                    logger.error(f"Invalid item structure in dimension: {dimension}")
                    return False
        
        return True


class PromptGenerator:
    """Generates model-specific prompts for different LLM architectures."""
    
    @staticmethod
    def create_model_prompt(statement: str, model_id: str) -> str:
        """
        Create appropriate prompt for specific model architecture.
        
        Args:
            statement: The privacy statement to evaluate
            model_id: Model identifier to determine prompt format
            
        Returns:
            Formatted prompt string for the model
        """
        instruction = (
            "Rate the following statement on a 7-point Likert scale. "
            "Only respond with a single number (1-7) where 1=strongly disagree "
            "and 7=strongly agree. Do not repeat the statement, explain your "
            "reasoning, or provide any additional text: "
        )
        
        model_lower = model_id.lower()
        
        if "deepseek" in model_lower:
            return f"<｜begin▁of▁sentence｜><｜User｜>{instruction}{statement}<｜Assistant｜>"
        elif "anthropic" in model_lower or "claude" in model_lower:
            return f"Human: {instruction}{statement}\n\nAssistant: "
        elif "meta" in model_lower or "llama" in model_lower:
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                f"{instruction}{statement}\n<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>"
            )
        elif "amazon" in model_lower or "titan" in model_lower:
            return f"{instruction}{statement}"
        else:
            return f"{instruction}{statement}"


class RequestBodyGenerator:
    """Generates model-specific request bodies for AWS Bedrock API calls."""
    
    @staticmethod
    def get_request_body(prompt: str, model_id: str) -> str:
        """
        Generate appropriate request body for specific model.
        
        Args:
            prompt: The formatted prompt
            model_id: Model identifier
            
        Returns:
            JSON string for the request body
        """
        model_lower = model_id.lower()
        
        if "anthropic" in model_lower or "claude" in model_lower:
            return json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "messages": [{"role": "user", "content": prompt}]
            })
        elif "meta" in model_lower or "llama" in model_lower:
            return json.dumps({
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.7,
                "top_p": 0.9
            })
        elif "amazon" in model_lower or "titan" in model_lower:
            return json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 512,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        else:
            return json.dumps({
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            })


class ResponseExtractor:
    """Extracts response text from different model output formats."""
    
    @staticmethod
    def extract_response_text(model_response: Dict[str, Any], model_id: str) -> str:
        """
        Extract response text from model output.
        
        Args:
            model_response: Raw response from the model
            model_id: Model identifier
            
        Returns:
            Extracted response text
        """
        model_lower = model_id.lower()
        
        try:
            if "anthropic" in model_lower or "claude" in model_lower:
                content = model_response.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "")
            elif "meta" in model_lower or "llama" in model_lower:
                return model_response.get("generation", "")
            elif "amazon" in model_lower or "titan" in model_lower:
                results = model_response.get("results", [])
                if len(results) > 0:
                    return results[0].get("outputText", "")
            else:
                choices = model_response.get("choices", [])
                if len(choices) > 0:
                    return choices[0].get("text", "")
                return model_response.get("completion", "")
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error extracting response text: {e}")
        
        return ""


class BedrockClientManager:
    """Manages AWS Bedrock client configuration and creation."""
    
    @staticmethod
    def create_client(model_id: str, config: EvaluationConfig) -> boto3.client:
        """
        Create configured Bedrock client with appropriate timeouts.
        
        Args:
            model_id: Model identifier for timeout configuration
            config: Evaluation configuration
            
        Returns:
            Configured boto3 Bedrock client
        """
        timeout_config = BedrockClientManager._get_timeout_config(model_id, config)
        
        try:
            client = boto3.client(
                "bedrock-runtime", 
                region_name=config.region_name,
                config=timeout_config
            )
            
            logger.info(
                f"Created Bedrock client for {model_id} with "
                f"read_timeout={timeout_config.read_timeout}s"
            )
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
    
    @staticmethod
    def _get_timeout_config(model_id: str, config: EvaluationConfig) -> Config:
        """
        Get timeout configuration for specific model.
        
        Args:
            model_id: Model identifier
            config: Evaluation configuration
            
        Returns:
            Boto3 Config object with timeout settings
        """
        # Check custom timeouts first
        if model_id in config.model_timeouts:
            read_timeout = config.model_timeouts[model_id]
        else:
            # Use default timeouts based on model family
            read_timeout = config.default_timeout
            for family, timeout in DEFAULT_MODEL_TIMEOUTS.items():
                if family in model_id.lower():
                    read_timeout = timeout
                    break
        
        return Config(
            read_timeout=read_timeout,
            connect_timeout=60,
            retries={'max_attempts': 10}
        )


class StanceAnalyzer:
    """Analyzes model responses to determine privacy stance."""
    
    def __init__(self):
        """Initialize stance analyzer with DSPy classifier."""
        self.classify_stance = dspy.Predict(LikertStance)
    
    def analyze_response(self, response_text: str) -> Dict[str, Union[str, int, float]]:
        """
        Analyze response text to determine stance and Likert score.
        
        Args:
            response_text: Raw response text from model
            
        Returns:
            Dictionary containing stance, likert_score, and confidence
        """
        try:
            # Use DSPy for classification
            classification = self.classify_stance(response=response_text)
            
            return {
                "stance": classification.stance,
                "likert_score": int(classification.likert_rating),
                "confidence": float(classification.confidence)
            }
            
        except Exception as e:
            logger.warning(f"DSPy classification failed: {e}")
            # Fallback to regex-based extraction
            return self._fallback_analysis(response_text)
    
    def _fallback_analysis(self, response_text: str) -> Dict[str, Union[str, int, float]]:
        """
        Fallback analysis using regex pattern matching.
        
        Args:
            response_text: Response text to analyze
            
        Returns:
            Dictionary containing stance analysis results
        """
        # Try to extract number directly
        number_match = re.search(r'\b[1-7]\b', response_text)
        
        if number_match:
            likert_score = int(number_match.group())
            stance = self._score_to_stance(likert_score)
            confidence = 0.7
        else:
            # Default values if no clear indication
            likert_score = 4
            stance = "NEUTRAL"
            confidence = 0.5
        
        return {
            "stance": stance,
            "likert_score": likert_score,
            "confidence": confidence
        }
    
    @staticmethod
    def _score_to_stance(score: int) -> str:
        """
        Convert Likert score to stance category.
        
        Args:
            score: Likert score (1-7)
            
        Returns:
            Stance category string
        """
        stance_mapping = {
            1: "STRONG_DISAGREE",
            2: "DISAGREE", 
            3: "SLIGHTLY_DISAGREE",
            4: "NEUTRAL",
            5: "SLIGHTLY_AGREE",
            6: "AGREE",
            7: "STRONG_AGREE"
        }
        return stance_mapping.get(score, "NEUTRAL")


class PrivacyEvaluator:
    """Main class for evaluating privacy attitudes of language models."""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize privacy evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.stance_analyzer = StanceAnalyzer()
        self.prompt_generator = PromptGenerator()
        self.request_generator = RequestBodyGenerator()
        self.response_extractor = ResponseExtractor()
    
    def evaluate_model(
        self, 
        model_id: str, 
        questionnaire: Dict[str, Any],
        dimensions: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single model's privacy attitudes.
        
        Args:
            model_id: AWS Bedrock model identifier
            questionnaire: Questionnaire data
            dimensions: Privacy dimensions to evaluate
            
        Returns:
            Evaluation results or None if evaluation fails
        """
        logger.info(f"Starting evaluation for model: {model_id}")
        
        if dimensions is None:
            dimensions = PRIVACY_DIMENSIONS
        
        try:
            client = BedrockClientManager.create_client(model_id, self.config)
        except Exception as e:
            logger.error(f"Failed to create client for {model_id}: {e}")
            return None
        
        results = {
            "model": model_id,
            "dimensions": {},
            "responses": {},
            "metadata": {
                "num_samples": self.config.num_samples,
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Evaluate each dimension
        for dimension in dimensions:
            if dimension not in questionnaire:
                logger.warning(f"Dimension {dimension} not found in questionnaire")
                continue
            
            logger.info(f"Evaluating {dimension} dimension")
            dimension_results = self._evaluate_dimension(
                client, model_id, dimension, questionnaire[dimension]
            )
            
            if dimension_results:
                results["dimensions"][dimension] = dimension_results["scores"]
                results["responses"][dimension] = dimension_results["responses"]
        
        # Calculate overall privacy score
        self._calculate_privacy_score(results)
        
        logger.info(f"Completed evaluation for {model_id}")
        return results
    
    def _evaluate_dimension(
        self, 
        client: boto3.client,
        model_id: str,
        dimension: str,
        dimension_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single privacy dimension.
        
        Args:
            client: Bedrock client
            model_id: Model identifier
            dimension: Dimension name
            dimension_data: Dimension questionnaire data
            
        Returns:
            Dimension evaluation results
        """
        dimension_scores = []
        dimension_responses = []
        
        items = dimension_data.get("items", [])
        
        for item in tqdm(items, desc=f"Processing {dimension}"):
            statement = item["text"]
            item_results = self._evaluate_statement(
                client, model_id, statement, item.get("id", "unknown")
            )
            
            if item_results:
                dimension_scores.append(item_results["avg_likert"])
                dimension_responses.append(item_results)
        
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
    
    def _evaluate_statement(
        self,
        client: boto3.client,
        model_id: str,
        statement: str,
        question_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single privacy statement with multiple samples.
        
        Args:
            client: Bedrock client
            model_id: Model identifier  
            statement: Privacy statement to evaluate
            question_id: Question identifier
            
        Returns:
            Statement evaluation results
        """
        sample_results = []
        
        for sample_num in range(self.config.num_samples):
            try:
                result = self._single_model_call(client, model_id, statement)
                if result:
                    sample_results.append(result)
                    logger.debug(
                        f"Sample {sample_num + 1}: {result['stance']} "
                        f"(score: {result['likert_score']}, "
                        f"confidence: {result['confidence']:.2f})"
                    )
                
                # Rate limiting between calls
                time.sleep(max(1, 1 * (1.5 ** sample_num) if sample_num < 5 else 5))
                
            except Exception as e:
                logger.warning(f"Sample {sample_num + 1} failed: {e}")
                continue
        
        if not sample_results:
            logger.error(f"All samples failed for statement: {statement[:50]}...")
            return None
        
        # Calculate average Likert score
        valid_scores = [r["likert_score"] for r in sample_results]
        avg_likert = sum(valid_scores) / len(valid_scores)
        
        return {
            "question_id": question_id,
            "question": statement,
            "samples": sample_results,
            "avg_likert": avg_likert,
            "num_valid_samples": len(valid_scores)
        }
    
    def _single_model_call(
        self,
        client: boto3.client,
        model_id: str,
        statement: str
    ) -> Optional[Dict[str, Any]]:
        """
        Make a single call to the model and analyze response.
        
        Args:
            client: Bedrock client
            model_id: Model identifier
            statement: Statement to evaluate
            
        Returns:
            Analysis results for single response
        """
        try:
            # Generate prompt and request body
            prompt = self.prompt_generator.create_model_prompt(statement, model_id)
            body = self.request_generator.get_request_body(prompt, model_id)
            
            # Make API call
            start_time = time.time()
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            response_time = time.time() - start_time
            
            # Parse response
            model_response = json.loads(response["body"].read())
            response_text = self.response_extractor.extract_response_text(
                model_response, model_id
            )
            
            # Analyze stance
            analysis = self.stance_analyzer.analyze_response(response_text)
            analysis["response_text"] = response_text
            analysis["response_time"] = response_time
            
            return analysis
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"AWS API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in model call: {e}")
            raise
    
    def _calculate_privacy_score(self, results: Dict[str, Any]) -> None:
        """
        Calculate overall privacy score from dimension scores.
        
        Args:
            results: Results dictionary to update with privacy score
        """
        required_dims = ["Control", "Awareness", "Collection"] 
        
        if not all(dim in results["dimensions"] for dim in required_dims):
            logger.warning("Cannot calculate privacy score: missing required dimensions")
            return
        
        control_score = results["dimensions"]["Control"]["average"]
        awareness_score = results["dimensions"]["Awareness"]["average"]
        collection_score = results["dimensions"]["Collection"]["average"]
        
        # Privacy score calculation (Collection is reverse-scored)
        privacy_score = (control_score + awareness_score + (8 - collection_score)) / 3
        results["privacy_score"] = privacy_score
        
        logger.info(f"Overall privacy score (1-7): {privacy_score:.2f}")


class ResultsSaver:
    """Handles saving evaluation results to files."""
    
    @staticmethod
    def save_results(results: Dict[str, Any], model_id: str, output_dir: str) -> None:
        """
        Save evaluation results to JSON files.
        
        Args:
            results: Evaluation results
            model_id: Model identifier
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate clean model name for filename
        model_name = ResultsSaver._clean_model_name(model_id)
        
        # Save detailed results
        detailed_path = os.path.join(
            output_dir, f"{model_name}_privacy_detailed_results.json"
        )
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to: {detailed_path}")
        
        # Save summary results
        summary = {
            "model": results["model"],
            "dimensions": {
                dim: data["average"] for dim, data in results["dimensions"].items()
            },
            "privacy_score": results.get("privacy_score"),
            "metadata": results.get("metadata", {})
        }
        
        summary_path = os.path.join(
            output_dir, f"{model_name}_privacy_summary_results.json"
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary results saved to: {summary_path}")
    
    @staticmethod
    def _clean_model_name(model_id: str) -> str:
        """
        Clean model ID for use in filenames.
        
        Args:
            model_id: Raw model identifier
            
        Returns:
            Cleaned model name suitable for filenames
        """
        if "/" in model_id:
            model_name = model_id.split("/")[-1].split(":")[0]
        else:
            model_parts = model_id.split(".")
            if len(model_parts) > 1:
                model_name = model_parts[1].split(":")[0]
            else:
                model_name = model_id
        
        return model_name.replace(".", "_").replace("-", "_")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate privacy attitudes of LLMs via AWS Bedrock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models
  python bedrock_privacy_evaluation.py --json_path questionnaire.json
  
  # Evaluate specific model
  python bedrock_privacy_evaluation.py --json_path questionnaire.json \\
    --single_model us.anthropic.claude-3-7-sonnet-20250219-v1:0
  
  # Custom timeout settings
  python bedrock_privacy_evaluation.py --json_path questionnaire.json \\
    --model_timeouts '{"us.anthropic.claude-3-7-sonnet-20250219-v1:0": 7200}'
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
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    parser.add_argument(
        '--region_name', 
        type=str, 
        default=DEFAULT_REGION,
        help=f'AWS region name (default: {DEFAULT_REGION})'
    )
    parser.add_argument(
        '--single_model', 
        type=str,
        help='Evaluate only this specific model ID'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=DEFAULT_SAMPLES,
        help=f'Number of samples per question (default: {DEFAULT_SAMPLES})'
    )
    parser.add_argument(
        '--default_timeout', 
        type=int, 
        default=DEFAULT_TIMEOUT,
        help=f'Default read timeout in seconds (default: {DEFAULT_TIMEOUT})'
    )
    parser.add_argument(
        '--model_timeouts', 
        type=str,
        help='JSON string with model-specific timeouts, e.g. \'{"model.id": 3600}\''
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
    
    # Validate environment
    if not EnvironmentValidator.validate_aws_credentials():
        logger.error("AWS credentials validation failed. Please check your configuration.")
        return
    
    if not EnvironmentValidator.validate_dspy_setup():
        logger.error("DSPy setup validation failed. Please check your OpenAI API key.")
        return
    
    # Parse model timeouts
    model_timeouts = {}
    if args.model_timeouts:
        try:
            model_timeouts = json.loads(args.model_timeouts)
            logger.info(f"Using custom model timeouts: {model_timeouts}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse model_timeouts JSON: {args.model_timeouts}")
    
    # Create configuration
    config = EvaluationConfig(
        region_name=args.region_name,
        num_samples=args.num_samples,
        default_timeout=args.default_timeout,
        model_timeouts=model_timeouts,
        output_dir=args.output_dir
    )
    
    # Load questionnaire
    questionnaire = QuestionnaireLoader.load_questionnaire(args.json_path)
    if not questionnaire:
        logger.error("Failed to load questionnaire. Exiting.")
        return
    
    # Determine models to evaluate
    if args.single_model:
        models = [args.single_model]
        logger.info(f"Evaluating single model: {args.single_model}")
    else:
        models = SUPPORTED_MODELS
        logger.info(f"Evaluating {len(models)} models")
    
    # Initialize evaluator
    evaluator = PrivacyEvaluator(config)
    
    # Evaluate each model
    successful_evaluations = 0
    for model_id in models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting evaluation: {model_id}")
        logger.info(f"{'='*80}")
        
        try:
            results = evaluator.evaluate_model(model_id, questionnaire)
            
            if results:
                ResultsSaver.save_results(results, model_id, config.output_dir)
                successful_evaluations += 1
                logger.info(f"✅ Successfully evaluated {model_id}")
            else:
                logger.error(f"❌ Failed to evaluate {model_id}")
                
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error evaluating {model_id}: {e}")
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation completed: {successful_evaluations}/{len(models)} models successful")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()