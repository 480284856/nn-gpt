"""
CyclicTrainingPipeline: Iterative LLM Fine-tuning with Vision Model Generation

This script implements a cyclic training pipeline that:
1. Uses LLM (gpt-oss-20b) to generate vision models via nn_gen
2. Evaluates code correctness and trains models on CIFAR-10 for 1 epoch
3. Collects accuracy metrics (Best, Average, Median)
4. Deduplicates generated code using nn-dup pipeline
5. Finetunes LLM with QLoRA using deduplicated data
6. Plots accuracy trends across all cycles

Output: A plot similar to "First-Epoch Accuracy Trends" showing Best/Avg/Median lines.

Usage:
    python -m ab.gpt.CyclicTrainingPipeline --num_cycles 22
    python -m ab.gpt.CyclicTrainingPipeline --num_cycles 5 --dry_run  # Test mode
"""

# Unsloth conditional import
try:
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import argparse
import json
import os
import sys
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# matplotlib is imported lazily in plot_accuracy_trends() to avoid startup errors

# Project imports
from ab.gpt.util.Const import nngpt_dir, conf_llm_dir, conf_test_dir
from ab.nn.util.Util import release_memory

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the cyclic training pipeline."""
    num_cycles: int = 22
    llm_conf: str = 'gpt_oss_20b.json'
    nn_gen_conf: str = 'NN_gen_novelty.json'
    nn_gen_conf_id: str = 'improve_classification_only'
    
    # Vision model training
    nn_train_epochs: int = 1
    test_nn: int = 10  # Number of vision models to generate per cycle
    
    # LLM fine-tuning (QLoRA)
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Output directories (follows nn-gpt Const.py style: nngpt_dir / llm / epoch / A{cycle})
    output_base: Path = field(default_factory=lambda: nngpt_dir)
    
    # Options
    dry_run: bool = False
    skip_nn_dup: bool = False
    
    # Improvement mode: Generate training data with paired models (reference + improved)
    # When True, uses JoinConf to fetch pairs of models with same conditions but different
    # accuracy, enabling training for "improve this model" task
    improvement_mode: bool = True

    # Debugging
    step4_test_prompt_num: int = -1
    
    def cycle_dir(self, cycle: int) -> Path:
        """Get output directory for a specific cycle. Style: nngpt_dir/llm/epoch/A{cycle}"""
        return self.output_base / 'llm' / 'epoch' / f'A{cycle}'
    
    def synth_dir(self, cycle: int) -> Path:
        """Get synthesized models directory for a cycle."""
        return self.cycle_dir(cycle) / 'synth_nn'
    
    def chat_data_dir(self, cycle: int) -> Path:
        """Get chat data directory for a cycle."""
        return self.cycle_dir(cycle) / 'chat_data'
    
    def lora_path(self, cycle: int) -> Optional[Path]:
        """Get LoRA checkpoint path from previous cycle."""
        if cycle == 0:
            return None
        prev_lora = self.cycle_dir(cycle - 1) / 'lora_checkpoint'
        return prev_lora if prev_lora.exists() else None


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class CycleMetrics:
    """Metrics collected from a single cycle."""
    cycle: int
    accuracies: List[float]
    num_generated: int
    num_valid: int
    num_evaluated: int
    
    @property
    def best(self) -> float:
        return max(self.accuracies) if self.accuracies else 0.0
    
    @property
    def average(self) -> float:
        return statistics.mean(self.accuracies) if self.accuracies else 0.0
    
    @property
    def median(self) -> float:
        return statistics.median(self.accuracies) if self.accuracies else 0.0


def collect_metrics_from_cycle(synth_dir: Path) -> Tuple[List[float], int, int]:
    """
    Collect accuracy metrics from eval_info.json files in synth_nn directory.
    
    Returns:
        Tuple of (accuracies list, num_valid models, num_evaluated models)
    """
    accuracies = []
    num_valid = 0
    num_evaluated = 0
    
    if not synth_dir.exists():
        print(f"  [WARN] Synth directory not found: {synth_dir}")
        return accuracies, num_valid, num_evaluated
    
    for model_dir in synth_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Check for new_nn.py (generated code)
        code_file = model_dir / 'new_nn.py'
        if code_file.exists():
            num_valid += 1
        
        # Check for eval_info.json (evaluation results)
        eval_file = model_dir / 'eval_info.json'
        if eval_file.exists():
            try:
                with open(eval_file) as f:
                    eval_data = json.load(f)
                
                # Extract accuracy from eval_results
                eval_results = eval_data.get('eval_results', {})
                if isinstance(eval_results, dict):
                    # Try different possible keys for accuracy
                    acc = eval_results.get('acc') or eval_results.get('accuracy') or eval_results.get('metric')
                    if acc is not None:
                        accuracies.append(float(acc))
                        num_evaluated += 1
                elif isinstance(eval_results, (int, float)):
                    accuracies.append(float(eval_results))
                    num_evaluated += 1
                elif isinstance(eval_results, list) and len(eval_results) >= 2:
                    # eval_results = [name, accuracy, accuracy_to_time, duration]
                    acc = eval_results[1]  # second element is accuracy
                    if acc is not None:
                        accuracies.append(float(acc))
                        num_evaluated += 1       
            except Exception as e:
                print(f"  [WARN] Failed to parse {eval_file}: {e}")
    
    return accuracies, num_valid, num_evaluated


# =============================================================================
# Visualization
# =============================================================================

def plot_accuracy_trends(metrics_history: List[CycleMetrics], save_path: Path):
    """
    Plot Best/Average/Median accuracy trends across cycles.
    
    Creates a plot similar to the reference "First-Epoch Accuracy Trends" image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot generation")
        return
    
    if not metrics_history:
        print("No metrics to plot")
        return
    
    cycles = [m.cycle for m in metrics_history]
    best_accs = [m.best * 100 for m in metrics_history]  # Convert to percentage
    avg_accs = [m.average * 100 for m in metrics_history]
    median_accs = [m.median * 100 for m in metrics_history]
    
    plt.figure(figsize=(12, 7))
    
    # Plot lines with markers (matching reference image style)
    plt.plot(cycles, best_accs, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Best')
    plt.plot(cycles, avg_accs, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Average')
    plt.plot(cycles, median_accs, '^-', color='#2ca02c', linewidth=2, markersize=6, label='Median')
    
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('First-Epoch Accuracy Trends', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Ensure integer x-ticks for cycles
    plt.xticks(cycles)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")


def save_metrics_json(metrics_history: List[CycleMetrics], save_path: Path):
    """Save metrics to JSON for later analysis."""
    data = []
    for m in metrics_history:
        data.append({
            'cycle': m.cycle,
            'best': m.best,
            'average': m.average,
            'median': m.median,
            'num_generated': m.num_generated,
            'num_valid': m.num_valid,
            'num_evaluated': m.num_evaluated,
            'accuracies': m.accuracies
        })
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Metrics saved to: {save_path}")

# =============================================================================
# function modification
# =============================================================================
from ab.nn.util.Util import uuid4
from ab.gpt.util.Util import read_py_file_as_string

import re
import ast
import ab.nn.api as api

def evaluate(self, nn_file):
    os.listdir(self.model_package)
    code = read_py_file_as_string(nn_file)
    if not code or not code.strip():
        raise Exception(f'Code is missing.')
    # sup_prm = 'supported_hyperparameters'
    # for fn in {sup_prm, 'train_setup', 'learn'}:
    #     if not re.match(r'[\s\S]*\s+def\s' + re.escape(fn) + r'\(.*', code):
    #         raise Exception(f'The NN code lacks the required function \'{fn}\'.')

    # tree = ast.parse(code)
    # for node in ast.walk(tree):
    #     if isinstance(node, ast.FunctionDef) and node.name == sup_prm:
    #         if isinstance(node.body[0], ast.Return):
    #             return_node = node.body[0].value
    #             prm_keys = ast.literal_eval(return_node)
    # for prm_key in prm_keys:
    #     if code.count('"' + prm_key + '"') + code.count("'" + prm_key + "'") < 2:
    #         raise Exception(f'The param \'{prm_key}\' is not used in the code.')

    api.data.cache_clear()
    df = api.data()
    ids_list = df["nn_id"].unique().tolist()
    new_checksum = uuid4(code)
    if new_checksum not in ids_list:
        return api.check_nn(code, self.task, self.dataset, self.metric, self.prm, self.save_to_db, self.prefix, self.save_path)
    else:
        # NN already exists, retrieve existing data from database
        print(f"  [WARNING] NN already exists (checksum: {new_checksum}). Returning cached results.")
        existing_row = df[df["nn_id"] == new_checksum].iloc[0]
        model_name = existing_row["nn"]
        accuracy = existing_row["accuracy"]
        # Return format matches api.check_nn: (model_name, accuracy, accuracy_to_time, score)
        return [model_name, accuracy, 0.0, 0.0]

from ab.gpt import NNEval


# =============================================================================
# Main Pipeline
# =============================================================================

class CyclicTrainingPipeline:
    """
    Main pipeline class that orchestrates all training cycles.
    
    Model is loaded ONCE at pipeline start and shared across all cycles.
    LoRA adapters are initialized ONCE and trained incrementally.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics_history: List[CycleMetrics] = []
        
        # Create output directory
        config.output_base.mkdir(parents=True, exist_ok=True)
        
        # Model instances (initialized in _load_model)
        self.model = None
        self.tokenizer = None
        self.chat_bot = None
        self.prompt_dict = None
        self.lora_tuner = None  # LoRA wrapper (initialized once in _init_lora)
        self.llm_config = None
        self._model_loaded = False
        self.dataset = 'cifar-10'
        self.epoch = 1

        # Debugging
        self.step4_test_prompt_num = self.config.step4_test_prompt_num
    
    def _load_model(self):
        """Load LLM once at pipeline start. Called from run()."""
        if self._model_loaded:
            return
        
        from ab.gpt.util.LLM import LLM
        from ab.gpt.util.Chatbot import ChatBot
        from ab.gpt.util.LLMUtil import quantization_config_4bit
        
        # Load LLM config
        with open(conf_llm_dir / self.config.llm_conf) as f:
            self.llm_config = json.load(f)
        
        base_model_name = self.llm_config['base_model_name']
        context_length = self.llm_config.get('context_length')
        use_unsloth = self.llm_config.get('use_unsloth', False)
        load_in_4bit = self.llm_config.get('load_in_4bit', True)
        
        print(f"  Loading LLM: {base_model_name}")
        model_loader = LLM(
            base_model_name,
            quantization_config_4bit,
            context_length=context_length,
            use_unsloth=use_unsloth,
            load_in_4bit=load_in_4bit
        )
        self.model = model_loader.get_model()
        self.tokenizer = model_loader.get_tokenizer()
        
        # Load prompt config for generation
        with open(conf_test_dir / self.config.nn_gen_conf) as f:
            self.prompt_dict = json.load(f)
        
        # NOTE: ChatBot is created in _init_lora after peft_model is ready
        
        self._model_loaded = True
        print(f"  Model loaded successfully")
    
    def _init_lora(self):
        """Initialize LoRA adapters ONCE. Called after _load_model in run()."""
        if self.lora_tuner is not None:
            return
        
        from peft import LoraConfig
        from transformers import TrainingArguments
        import torch
        from ab.gpt.util.LoRA import LoRA
        
        use_unsloth = self.llm_config.get('use_unsloth', False)
        
        # Configure training arguments (will be reused/updated per cycle)
        training_args = TrainingArguments(
            output_dir=str(self.config.output_base / 'lora_training'),
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            logging_steps=10,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            optim='paged_adamw_8bit',
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            report_to=[],
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'),
            lora_dropout=self.config.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        
        print(f"  Initializing LoRA (r={self.config.r}, alpha={self.config.lora_alpha})")
        # LoRA.__init__ attaches adapters to model
        self.lora_tuner = LoRA(
            self.model,
            self.tokenizer,
            training_args=training_args,
            peft_config=peft_config,
            use_unsloth=use_unsloth
        )
        print(f"  LoRA initialized successfully")
        
        # Create ChatBot with peft_model (after LoRA adapters are attached)
        from ab.gpt.util.Chatbot import ChatBot
        self.chat_bot = ChatBot(self.lora_tuner.peft_model, self.tokenizer, temperature=0.8, top_k=70, top_p=0.9)
    
    def _cleanup(self):
        """Release GPU memory at pipeline end."""
        print("  Cleaning up GPU memory...")
        self.model = None
        self.tokenizer = None
        self.chat_bot = None
        self.lora_tuner = None
        release_memory()
    
    def _step1_generate(self, cycle: int) -> Path:
        """Step 1: Generate vision models using shared LLM instance."""
        print(f"\n{'='*60}")
        print(f"STEP 1: Generate Vision Models (Cycle {cycle})")
        print(f"{'='*60}")
        
        synth_dir = self.config.synth_dir(cycle)
        synth_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Would generate {self.config.test_nn} vision models")
            return synth_dir
        
        # Use shared chat_bot for generation
        # self._nn_gen(cycle, self.dataset, self.epoch)
        return synth_dir
    
    def _nn_gen(self, cycle: int, dataset: str, epoch: int) -> int:
        """
        Generate vision models for a specific dataset using shared LLM instance.
        
        Args:
            cycle: The current cycle number.
            dataset: The dataset to generate models for.
        
        Returns:
            The number of models generated.
        """
        from tqdm import tqdm
        import ab.nn.api as api
        from ab.gpt.util.Const import new_nn_file, hp_file, transformer_file, new_out_file
        from ab.nn.util.Util import create_file
        
        synth_dir = self.config.synth_dir(cycle)
        max_new_tokens = self.llm_config.get('max_new_tokens', 3000)
        unsloth_max_input_length = self.llm_config.get('max_input_length', None)
        
        # Prepare prompts
        print(f"  Preparing prompts for {self.config.test_nn} models...")
        prompts = []
        conf_key = self.config.nn_gen_conf_id
        key_dict = self.prompt_dict[conf_key]
        
        prompt_template = '\n'.join(key_dict['prompt'])
        
        if not self.config.dry_run:
            # Get nn-dataset codes
            data = api.data(only_best_accuracy=False, task=key_dict['task'], dataset=dataset, epoch=epoch).groupby(by='nn').sample(n=1)
            bar = tqdm(data.iterrows(), total=len(data), desc="Preparing prompts")
            for _, row in bar:
                para_dict = {}
                for it in key_dict['input_list']:
                    para_dict[it['para']] = row[it['value']]
                prompt = prompt_template.format(**para_dict)
                # For infering with gpt-oss 20b in a single 4090 GPU,
                # we need to limit the prompt length to 1600 tokens to avoid OOM.
                if unsloth_max_input_length: # if the variable is set, it means we need to limit the prompt length
                    in_text = [{"role": "user", "content": prompt}]
                    output = self.chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
                    if len(output) <= unsloth_max_input_length:
                        prompts.append((prompt, row))
                    if len(prompts) >= self.config.test_nn:
                        break
        else:
            prompts = ["say 1+1=2"] * self.config.test_nn
        
       
        
        if self.config.test_nn: # limit the number of prompts for testing at inference time
            if len(prompts) > self.config.test_nn:
                prompts = prompts[:self.config.test_nn]
            else:
                print(f"[WARN] Not enough prompts to test, using {len(prompts)} prompts")
                

        # Generate models
        print(f"  Generating {len(prompts)} vision models...")
        generated_count = 0
        for idx, (prompt, origdf) in enumerate(tqdm(prompts, desc="Generating")):
            model_dir = synth_dir / f'B{idx}'
            
            # Generate
            if not self.config.dry_run:
                code, hp, tr, full_out = self.chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)
            else:
                code = "test: print('1+1=2')"
                hp = "{}"
                tr = "test: print('1+1=2')"
                full_out = "test: 1+1=2"
            
            # Save outputs
            model_dir.mkdir(parents=True, exist_ok=True)
            try:
                hp_parsed = json.loads(hp.replace("'", '"'))
                with open(model_dir / hp_file, 'w') as f:
                    json.dump(hp_parsed, f)
            except Exception as e:
                print(f"  [WARN] Failed to parse HP: {e}")
                continue
            
            try:
                create_file(model_dir, transformer_file, tr)
            except:
                pass
            
            create_file(model_dir, new_nn_file, code)
            create_file(model_dir, new_out_file, full_out)
            
            # Save original df info
            if origdf is not None:
                create_file(model_dir, f"original_{origdf['nn']}.py", origdf['nn_code'])
                origdf.to_pickle(model_dir / 'dataframe.df')
            
            generated_count += 1
        
        print(f"  Generated {generated_count} models")
        api.data.cache_clear()
        return generated_count
    
    def _step2_evaluate(self, cycle: int, synth_dir: Path) -> CycleMetrics:
        """Step 2: Evaluate generated models on CIFAR-10."""
        print(f"\n{'='*60}")
        print(f"STEP 2: Evaluate Models on CIFAR-10 (Cycle {cycle})")
        print(f"{'='*60}")
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Would evaluate models in: {synth_dir}")
            dummy_accs = [0.3 + 0.02 * cycle + 0.1 * i / 10 for i in range(10)]
            return CycleMetrics(
                cycle=cycle,
                accuracies=dummy_accs,
                num_generated=10,
                num_valid=10,
                num_evaluated=10
            )
         
        print(f"  Running NNEval on {synth_dir}...")
        
        # Patch Eval.evaluate to handle duplicates gracefully (return cached results instead of raising exception)
        from ab.gpt.util.Eval import Eval
        Eval.evaluate = evaluate
        
        NNEval.main(
            nn_name_prefix=f"cyc{cycle}",
            nn_train_epochs=self.config.nn_train_epochs,
            save_to_db=True,
            cycle=cycle,
            nn_alter_epochs=1,
            custom_synth_dir=str(synth_dir)
        )
        
        # Collect metrics
        accuracies, num_valid, num_evaluated = collect_metrics_from_cycle(synth_dir)
        
        metrics = CycleMetrics(
            cycle=cycle,
            accuracies=accuracies,
            num_generated=len(list(synth_dir.iterdir())) if synth_dir.exists() else 0,
            num_valid=num_valid,
            num_evaluated=num_evaluated
        )
        
        print(f"  Results: {num_evaluated} models evaluated")
        print(f"  Best: {metrics.best:.4f}, Avg: {metrics.average:.4f}, Median: {metrics.median:.4f}")
        
        release_memory()
        return metrics
    
    def _step3_deduplicate(self, cycle: int) -> Path:
        """Step 3: Deduplicate generated code using nn-dup."""
        print(f"\n{'='*60}")
        print(f"STEP 3: Deduplicate Code with nn-dup (Cycle {cycle})")
        print(f"{'='*60}")
        
        chat_data_dir = self.config.chat_data_dir(cycle)
        chat_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Import nn-dup modules
        try:
            from ab.dup.preprocessing import curate_from_lemur
            from ab.chatprep import ChatPrepConfig
        except ImportError as e:
            print(f"  [WARN] nn-dup not available: {e}")
            print(f"  Falling back to using existing LEMUR data...")
            return chat_data_dir
        
        # Step 3a: Run deduplication
        dedup_output = self.config.cycle_dir(cycle) / 'dedup_output'
        dedup_output.mkdir(parents=True, exist_ok=True)
        
        print(f"  Running deduplication pipeline...")
        import logging
        logger = logging.getLogger('nn-dup')
        logger.setLevel(logging.INFO)
        
        curate_from_lemur(
            out_dir=dedup_output,
            min_per_prefix=1,
            keep_per_family=5,
            cache_dir=None,
            logger=logger,
            improvement_mode=self.config.improvement_mode,
            epoch=self.epoch
        )
        
        # Step 3b: Convert to chat format
        accepted_code_dir = dedup_output / 'accepted_code'
        if accepted_code_dir.exists() and any(accepted_code_dir.iterdir()):
            print(f"  Running ChatPrep...")
            chat_config = ChatPrepConfig(
                accepted_dir=str(accepted_code_dir),
                out_dir=str(chat_data_dir),
                seed=42 + cycle,
                fix_fences=True,
                drop_unparseable=True,
                improvement_mode=self.config.improvement_mode
            )
            chat_config.run()
        else:
            print(f"  [WARN] No accepted code found, skipping ChatPrep")
        
        return chat_data_dir
    
    def _step4_finetune(self, cycle: int, chat_data_dir: Path, test_prompt_num: int=-1):
        """
        Step 4: Fine-tune using shared LoRA tuner, save checkpoint.

        Args:
            cycle (int): The current cycle number.
            chat_data_dir (Path): The directory containing the chat data.
            test_prompt_num (int): [Only for debugging] The number of test prompts to use.
        """
        print(f"\n{'='*60}")
        print(f"STEP 4: Fine-tune LLM with QLoRA (Cycle {cycle})")
        print(f"{'='*60}")
        
        lora_output = self.config.cycle_dir(cycle) / 'lora_checkpoint'
        lora_output.mkdir(parents=True, exist_ok=True)
        
        if self.config.dry_run:
            print(f"  [DRY RUN] Skipping fine-tune LLM")
            return
        
        # Load training data from JSONL files
        jsonl_files = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
        all_data = []
        for jsonl_name in jsonl_files:
            jsonl_file = chat_data_dir / jsonl_name
            if jsonl_file.exists():
                print(f"  Loading {jsonl_name}...")
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
        
        if not all_data:
            print(f"  [WARN] No training data found, skipping fine-tuning")
            return
        
        if test_prompt_num >= 0:
            all_data = all_data[:test_prompt_num]
        print(f"  Loaded {len(all_data)} examples")
        
        # Convert to text format
        from datasets import Dataset
        text_data = []
        for item in all_data:
            messages = item.get('messages', [])
            if messages:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                text_data.append({'text': text})
        
        if not text_data:
            print(f"  [WARN] No valid training examples")
            return
        
        dataset = Dataset.from_list(text_data)
        print(f"  Training on {len(dataset)} examples...")
        
        # Train using shared lora_tuner (LoRA already attached)
        self.lora_tuner.peft_model.train()
        self.lora_tuner.train(dataset, self.tokenizer, str(lora_output))
        
        print(f"  LoRA checkpoint saved to: {lora_output}")
    
    def run_cycle(self, cycle: int) -> CycleMetrics:
        """Run a single training cycle using shared model instance."""
        print(f"\n{'#'*70}")
        print(f"# CYCLE {cycle}/{self.config.num_cycles}")
        print(f"{'#'*70}")
        
        # Step 1: Generate vision models (uses shared chat_bot)
        synth_dir = self._step1_generate(cycle)
        
        # # Offload model to CPU to free GPU memory for Step 2
        # if not self.config.dry_run and self.lora_tuner is not None:
        #     print("  Offloading LLM to CPU...")
        #     self.lora_tuner.peft_model.to('cpu')
        #     release_memory()
        
        # # Step 2: Evaluate models on CIFAR-10
        metrics = self._step2_evaluate(cycle, synth_dir)
        
        # # Move model back to GPU after Step 2
        # if not self.config.dry_run and self.lora_tuner is not None:
        #     print("  Moving LLM back to GPU...")
        #     self.lora_tuner.peft_model.to('cuda')
        
        # Step 3: Deduplicate code with nn-dup
        chat_data_dir = self._step3_deduplicate(cycle)
        
        # Step 4: Fine-tune LLM with QLoRA (uses shared lora_tuner)
        self._step4_finetune(cycle, chat_data_dir, self.step4_test_prompt_num)
        
        return metrics

    def run(self):
        """Run all training cycles with single model instance."""
        print(f"\n{'='*70}")
        print("CYCLIC TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Number of cycles: {self.config.num_cycles}")
        print(f"LLM config: {self.config.llm_conf}")
        print(f"Output directory: {self.config.output_base}")
        print(f"Dry run: {self.config.dry_run}")
        
        # Load model ONCE
        if not self.config.dry_run:
            self._load_model()
            self._init_lora()
        
        for cycle in range(self.config.num_cycles):
            try:
                metrics = self.run_cycle(cycle)
                self.metrics_history.append(metrics)
                
                print(f"\n[Cycle {cycle}] Best={metrics.best:.4f}, Avg={metrics.average:.4f}, Med={metrics.median:.4f}")
                self.save_results()
                
            except Exception as e:
                print(f"\n[ERROR] Cycle {cycle} failed: {e}")
                import traceback
                traceback.print_exc()
                
                self.metrics_history.append(CycleMetrics(
                    cycle=cycle,
                    accuracies=[],
                    num_generated=0,
                    num_valid=0,
                    num_evaluated=0
                ))
        
        # Cleanup
        if not self.config.dry_run:
            self._cleanup()
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save current results (plot and JSON)."""
        plot_path = self.config.output_base / 'accuracy_trends.png'
        json_path = self.config.output_base / 'metrics.json'
        
        plot_accuracy_trends(self.metrics_history, plot_path)
        save_metrics_json(self.metrics_history, json_path)
    
    def print_summary(self):
        """Print final summary of all cycles."""
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        
        print(f"\nCycle | Best   | Average | Median | Evaluated")
        print("-" * 50)
        for m in self.metrics_history:
            print(f"  {m.cycle:2d}  | {m.best:.4f} | {m.average:.4f}  | {m.median:.4f} | {m.num_evaluated}")
        
        if self.metrics_history:
            final = self.metrics_history[-1]
            first = self.metrics_history[0]
            print(f"\nImprovement (Cycle 0 → {final.cycle}):")
            print(f"  Best:    {first.best:.4f} → {final.best:.4f} ({(final.best - first.best)*100:+.2f}%)")
            print(f"  Average: {first.average:.4f} → {final.average:.4f} ({(final.average - first.average)*100:+.2f}%)")
            print(f"  Median:  {first.median:.4f} → {final.median:.4f} ({(final.median - first.median)*100:+.2f}%)")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cyclic Training Pipeline: LLM generates vision models, evaluates them, and improves via QLoRA.'
    )
    
    # Main parameters
    parser.add_argument('--num_cycles', type=int, default=22,
                        help='Number of training cycles (default: 22)')
    parser.add_argument('--llm_conf', type=str, default='gpt_oss_20b.json',
                        help='LLM configuration file (default: gpt_oss_20b.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: out/cyclic_pipeline)')
    
    # Generation parameters
    parser.add_argument('--test_nn', type=int, default=10,
                        help='Number of vision models to generate per cycle (default: 10)')
    parser.add_argument('--nn_train_epochs', type=int, default=1,
                        help='Epochs to train each vision model (default: 1)')
    
    # LLM training parameters
    parser.add_argument('--num_train_epochs', type=int, default=1,
                        help='LLM fine-tuning epochs per cycle (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for QLoRA (default: 1e-5)')
    parser.add_argument('--r', type=int, default=32,
                        help='LoRA rank (default: 32)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (default: 32)')
    
    # Options
    parser.add_argument('--dry_run', action='store_true',
                        help='Dry run mode (no actual training)')
    parser.add_argument('--skip_nn_dup', action='store_true',
                        help='Skip nn-dup deduplication step')
    
    # Debugging
    parser.add_argument('--step4_test_prompt_num', type=int, default=-1,
                        help='[Only for debugging] Number of test prompts to use in step 4 (default: -1, use all)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = PipelineConfig(
        num_cycles=args.num_cycles,
        llm_conf=args.llm_conf,
        test_nn=args.test_nn,
        nn_train_epochs=args.nn_train_epochs,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        r=args.r,
        lora_alpha=args.lora_alpha,
        dry_run=args.dry_run,
        skip_nn_dup=args.skip_nn_dup,
        step4_test_prompt_num=args.step4_test_prompt_num
    )
    
    if args.output:
        config.output_base = Path(args.output)
    
    # Run pipeline
    pipeline = CyclicTrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
