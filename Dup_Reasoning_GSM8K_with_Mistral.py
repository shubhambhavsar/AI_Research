import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from dotenv import load_dotenv
from datasets import load_dataset
from fractions import Fraction
import re

# -----------------------------
# Mistral API
# -----------------------------
from mistralai import Mistral

# -----------------------------
# Model configuration
# -----------------------------
DUP_REASONING_MODEL = "open-mixtral-8x22b"   # Stages 1–3 + CoT reasoning
ANSWER_EXTRACTION_MODEL = "open-mixtral-8x7b"  # JSON-only numeric extraction

# Load environment variables
load_dotenv()

@dataclass
class GSM8KExample:
    question: str
    answer: str
    numerical_answer: float

@dataclass
class DUPResult:
    original_question: str
    ground_truth_answer: str
    ground_truth_numerical: float
    core_question: str
    problem_solving_info: str
    final_response: str
    final_numerical_answer: Optional[float]
    zero_shot_cot_response: str
    zero_shot_cot_numerical: Optional[float]
    dup_correct: bool
    zero_shot_cot_correct: bool
    processing_time: float
    tokens_used: int

class GSM8KDataLoader:
    """Load and sample GSM8K dataset from Hugging Face"""

    def __init__(self, api_key: Optional[str] = None):
        self.dataset = None
        self.client = Mistral(api_key=api_key) if api_key else None

    def load_sample_data(self, n_samples: int = 50, split: str = "test") -> List[GSM8KExample]:
        print("Loading GSM8K dataset from Hugging Face...")
        try:
            self.dataset = load_dataset("openai/gsm8k", "main")
            print("Successfully loaded GSM8K dataset")
        except Exception as e:
            raise Exception(f"Failed to load GSM8K dataset: {e}")

        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            raise ValueError(f"Split '{split}' not available. Available splits: {available_splits}")

        data = self.dataset[split]
        print(f"Dataset split '{split}' contains {len(data)} examples")

        total_examples = len(data)
        n_samples = min(n_samples, total_examples)

        random.seed(123)  # reproducible sample
        sample_indices = random.sample(range(total_examples), n_samples)
        print(f"Sampling {n_samples} examples from {total_examples} total examples")

        examples = []
        for idx in sample_indices:
            item = data[idx]
            numerical_answer = self._extract_numerical_answer_llm(item['answer'])
            examples.append(GSM8KExample(
                question=item['question'],
                answer=item['answer'],
                numerical_answer=numerical_answer
            ))

        print(f"Successfully loaded {len(examples)} examples")
        return examples

    def _extract_numerical_answer_llm(self, answer_text: str) -> float:
        # """Use the LLM to extract the ground-truth numeric answer from GSM8K's answer text.
        # Returns 0.0 on failure to keep typing stable for downstream code."""
        # if self.client is None:
        #     return 0.0

        # prompt = (
        #     "You are given a GSM8K answer/explanation. Identify the single FINAL numeric answer.\n"
        #     "Rules:\n"
        #     "- Respond in STRICT JSON only: {\"value\": <number or null>}.\n"
        #     "- 'value' must be a number (int or float) without units. Do not include any text.\n"
        #     "- If the final value is a fraction (e.g., 3/4), convert it to decimal (0.75).\n"
        #     "- Ignore currency symbols, percent signs, or commas.\n"
        #     "- If you truly cannot find the final numeric answer, use null.\n\n"
        #     f"Text:\n{answer_text}"
        # )
        # try:
        #     resp = self.client.chat.complete(
        #         model=ANSWER_EXTRACTION_MODEL,
        #         messages=[{"role": "user", "content": prompt}],
        #         temperature=0,
        #         max_tokens=64,
        #     )
        #     content = resp.choices[0].message.content.strip()
        #     if content.startswith("```"):
        #         lines = content.splitlines()
        #         if lines and lines[0].startswith("```"):
        #             lines = lines[1:]
        #         if lines and lines[-1].startswith("```"):
        #             lines = lines[:-1]
        #         content = "\n".join(lines).strip()
        #     obj = json.loads(content)
        #     val = obj.get("value", None)
        #     if isinstance(val, (int, float)):
        #         return float(val)
        #     if isinstance(val, str):
        #         s = val.replace(",", "").strip()
        #         try:
        #             return float(s)
        #         except:
        #             try:
        #                 return float(Fraction(s))
        #             except:
        #                 return 0.0
        #     return 0.0
        # except Exception:
        #     return 0.0
        """Extract numerical answer from answer text"""
        # Look for #### pattern common in GSM8K
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
        if match:
            return float(match.group(1))
        
        # Fallback: look for last number in the text
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return float(numbers[-1])
        
        return 0.0

class DUPSystem:
    """Implementation of the DUP method"""

    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

        self.stage1_prompt = "Please extract core question, only the most comprehensive and detailed one!"
        self.stage2_prompt_template = (
            "Note: Please extract the problem-solving information related to the core question "
            "({core_question}), only extract the most useful information, list them one by one!"
        )
        self.stage3_prompt_template = """Hint: {problem_solving_info}
{core_question}
Please understand the Hint and question information, then solve the question step by step and show the answer."""
        self.zero_shot_cot_prompt = "Let's think step by step."
        self.answer_extraction_prompt = "Please extract the final numerical answer from the above solution. Respond with only the number."

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the reasoning model (22B) with retry logic"""
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.complete(
                    model=DUP_REASONING_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
        return ""

    def _llm_extract_numeric(self, text: str) -> Optional[float]:
        """Ask the extraction model (7B) to return {"value": <number or null>}"""
        prompt = (
            "Extract the single FINAL numeric answer from the text below.\n"
            "Rules:\n"
            "- Respond in STRICT JSON only: {\"value\": <number or null>}.\n"
            "- 'value' must be a number (int or float), without units.\n"
            "- If the final value is a fraction like 3/4, convert to decimal (0.75).\n"
            "- Ignore commas and symbols.\n"
            "- If you cannot find a final numeric answer, use null.\n\n"
            f"Text:\n{text}"
        )
        try:
            resp = self.client.chat.complete(
                model=ANSWER_EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            obj = json.loads(content)
            val = obj.get("value", None)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                s = val.replace(",", "").strip()
                try:
                    return float(s)
                except:
                    try:
                        return float(Fraction(s))
                    except:
                        return None
            return None
        except Exception:
            return None

    def extract_numerical_answer(self, text: str) -> Optional[float]:
        return self._llm_extract_numeric(text)

    def run_dup_method(self, question: str) -> Dict[str, Any]:
        start_time = time.time()

        stage1_input = f"{question}\n{self.stage1_prompt}"
        core_question = self._call_llm(stage1_input)

        stage2_prompt = self.stage2_prompt_template.format(core_question=core_question)
        stage2_input = f"{question}\n{stage2_prompt}"
        problem_solving_info = self._call_llm(stage2_input)

        stage3_prompt = self.stage3_prompt_template.format(
            problem_solving_info=problem_solving_info,
            core_question=core_question
        )
        stage3_input = f"{question}\n{stage3_prompt}"
        final_response = self._call_llm(stage3_input)

        processing_time = time.time() - start_time

        return {
            'core_question': core_question,
            'problem_solving_info': problem_solving_info,
            'final_response': final_response,
            'processing_time': processing_time
        }

    def run_zero_shot_cot(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        prompt = f"{question}\n{self.zero_shot_cot_prompt}"
        response = self._call_llm(prompt)
        processing_time = time.time() - start_time
        return {'response': response, 'processing_time': processing_time}

    def evaluate_single_example(self, example: GSM8KExample) -> DUPResult:
        print(f"Processing: {example.question[:100]}...")

        dup_results = self.run_dup_method(example.question)
        dup_numerical = self.extract_numerical_answer(dup_results['final_response'])

        cot_results = self.run_zero_shot_cot(example.question)
        cot_numerical = self.extract_numerical_answer(cot_results['response'])

        dup_correct = (dup_numerical is not None and
                       abs(dup_numerical - example.numerical_answer) < 0.01)
        cot_correct = (cot_numerical is not None and
                       abs(cot_numerical - example.numerical_answer) < 0.01)

        return DUPResult(
            original_question=example.question,
            ground_truth_answer=example.answer,
            ground_truth_numerical=example.numerical_answer,
            core_question=dup_results['core_question'],
            problem_solving_info=dup_results['problem_solving_info'],
            final_response=dup_results['final_response'],
            final_numerical_answer=dup_numerical,
            zero_shot_cot_response=cot_results['response'],
            zero_shot_cot_numerical=cot_numerical,
            dup_correct=dup_correct,
            zero_shot_cot_correct=cot_correct,
            processing_time=dup_results['processing_time'] + cot_results['processing_time'],
            tokens_used=0
        )

class GSM8KEvaluator:
    """Evaluate and analyze results"""

    def __init__(self):
        self.results = []

    def run_evaluation(self, n_samples: int = 20, split: str = "test"):
        print(f"Starting GSM8K evaluation with {n_samples} samples from '{split}' split...")

        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        data_loader = GSM8KDataLoader(api_key=api_key)
        dup_system = DUPSystem(api_key)

        print("Loading GSM8K dataset...")
        examples = data_loader.load_sample_data(n_samples, split=split)

        results = []
        for i, example in enumerate(examples, 1):
            print(f"\nProcessing example {i}/{len(examples)}")
            try:
                result = dup_system.evaluate_single_example(example)
                results.append(result)

                dup_acc = sum(r.dup_correct for r in results) / len(results)
                cot_acc = sum(r.zero_shot_cot_correct for r in results) / len(results)
                print(f"Current DUP accuracy: {dup_acc:.2%}")
                print(f"Current CoT accuracy: {cot_acc:.2%}")

            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue

        self.results = results
        return results

    def save_results_to_csv(self, filename: str = "gsm8k_dup_evaluation.csv"):
        if not self.results:
            print("No results to save")
            return

        csv_data = []
        for result in self.results:
            csv_data.append({
                'question': result.original_question,
                'core_question': result.core_question,
                'problem_solving_info': result.problem_solving_info,
                'dup_final_response': result.final_response,
                'dup_numerical_answer': result.final_numerical_answer,
                'dup_correct': result.dup_correct,
                'zero_shot_cot_response': result.zero_shot_cot_response,
                'zero_shot_cot_numerical': result.zero_shot_cot_numerical,
                'zero_shot_cot_correct': result.zero_shot_cot_correct,
                'ground_truth_answer': result.ground_truth_numerical,
                'processing_time': result.processing_time
            })

        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df

    def print_summary_statistics(self):
        if not self.results:
            print("No results to analyze")
            return

        n_total = len(self.results)
        dup_correct = sum(r.dup_correct for r in self.results)
        cot_correct = sum(r.zero_shot_cot_correct for r in self.results)

        dup_accuracy = dup_correct / n_total
        cot_accuracy = cot_correct / n_total
        avg_processing_time = sum(r.processing_time for r in self.results) / n_total

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total examples evaluated: {n_total}")
        print(f"DUP Method accuracy: {dup_accuracy:.2%} ({dup_correct}/{n_total})")
        print(f"Zero-shot CoT accuracy: {cot_accuracy:.2%} ({cot_correct}/{n_total})")
        print(f"Improvement: {dup_accuracy - cot_accuracy:.2%}")
        print(f"Average processing time: {avg_processing_time:.2f} seconds")

def main():
    """Main function to run the evaluation"""
    N_SAMPLES = 150
    SPLIT = "test"

    try:
        evaluator = GSM8KEvaluator()
        _ = evaluator.run_evaluation(n_samples=N_SAMPLES, split=SPLIT)
        _ = evaluator.save_results_to_csv()
        evaluator.print_summary_statistics()
        print("\nEvaluation complete! Results saved to CSV.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()