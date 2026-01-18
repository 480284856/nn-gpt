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
from ab.gpt.util.Const import nngpt_dir, out_dir, conf_llm_dir, conf_test_dir
from ab.nn.util.Util import release_memory


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the cyclic training pipeline."""
    num_cycles: int = 22
    llm_conf: str = 'gpt_oss_20b.json'
    nn_gen_conf: str = 'NN_gen.json'
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
    
    # Output directories
    output_base: Path = field(default_factory=lambda: out_dir / 'cyclic_pipeline')
    
    # Options
    dry_run: bool = False
    skip_nn_dup: bool = False
    
    def cycle_dir(self, cycle: int) -> Path:
        """Get output directory for a specific cycle."""
        return self.output_base / f'cycle_{cycle}'
    
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
                    
            except Exception as e:
                print(f"  [WARN] Failed to parse {eval_file}: {e}")
    
    return accuracies, num_valid, num_evaluated


# =============================================================================
# Pipeline Steps
# =============================================================================

def nn_gen(config: PipelineConfig, cycle: int, chat_bot, prompt_dict: dict) -> int:
    """
    Generate vision models using LLM (without calling NNEval).
    
    This is a custom implementation that does NOT automatically call NNEval.main(),
    allowing us to control the evaluation step separately with correct parameters.
    
    Args:
        config: Pipeline configuration
        cycle: Current cycle number
        chat_bot: ChatBot instance for generation
        prompt_dict: Prompt configuration dictionary
        
    Returns:
        Number of models successfully generated
    """
    from tqdm import tqdm
    import ab.nn.api as lemur
    from ab.gpt.util.Const import new_nn_file, hp_file, transformer_file, new_out_file
    from ab.nn.util.Util import create_file
    
    synth_dir = config.synth_dir(cycle)
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    # Get LLM config for max tokens
    with open(conf_llm_dir / config.llm_conf) as f:
        llm_config = json.load(f)
    max_new_tokens = llm_config.get('max_new_tokens', 3000)
    unsloth_max_input_length = llm_config.get('max_input_length', None)
    
    # Prepare prompts
    print(f"  Preparing prompts for {config.test_nn} models...")
    prompts = []
    conf_key = config.nn_gen_conf_id
    key_dict = prompt_dict[conf_key]
    
    prompt_template = '\n'.join(key_dict['prompt'])
    
    # Get nn-dataset codes
    data = lemur.data(only_best_accuracy=True, task=key_dict['task']).groupby(by='nn').sample(n=1)[:config.test_nn]
    addon_data = lemur.data(only_best_accuracy=True, task=key_dict.get('addon_task', key_dict['task']))
    
    for _, row in data.iterrows():
        para_dict = {}
        for it in key_dict['input_list']:
            para_dict[it['para']] = row[it['value']]
        # Avoid sampling same nn_code
        addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
        if key_dict.get('addon_list'):
            for it in key_dict['addon_list']:
                para_dict[it['para']] = addon_row[it['value']]
        prompts.append((prompt_template.format(**para_dict), row))
    
    # Generate models
    print(f"  Generating {len(prompts)} vision models...")
    generated_count = 0
    for idx, (prompt, origdf) in enumerate(tqdm(prompts, desc="Generating")):
        model_dir = synth_dir / f'B{idx}'
        
        # Check prompt length
        if unsloth_max_input_length:
            in_text = [{"role": "user", "content": prompt}]
            output = chat_bot.tokenizer.apply_chat_template(in_text, add_generation_prompt=True)
            if len(output) > unsloth_max_input_length:
                print(f'  Prompt too long ({len(output)} > {unsloth_max_input_length}), skipping...')
                continue
        
        # Generate
        code, hp, tr, full_out = chat_bot.chat(prompt, engineer_prompt=False, max_new_tokens=max_new_tokens)
        
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
    
    # Clear LEMUR cache
    lemur.data.cache_clear()
    
    return generated_count


def step1_generate_vision_models(config: PipelineConfig, cycle: int, lora_path: Optional[Path]) -> Path:
    """
    Step 1: Generate vision models using the LLM.
    
    Uses custom nn_gen that does NOT call NNEval.main().
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Generate Vision Models (Cycle {cycle})")
    print(f"{'='*60}")
    
    synth_dir = config.synth_dir(cycle)
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    if config.dry_run:
        print(f"  [DRY RUN] Would generate {config.test_nn} vision models")
        print(f"  [DRY RUN] Output dir: {synth_dir}")
        return synth_dir
    
    # Load LLM
    from ab.gpt.util.LLM import LLM
    from ab.gpt.util.Chatbot import ChatBot
    from ab.gpt.util.LLMUtil import quantization_config_4bit
    from ab.gpt.util.Const import conf_test_dir
    from peft import PeftModel
    
    with open(conf_llm_dir / config.llm_conf) as f:
        llm_config = json.load(f)
    
    base_model_name = llm_config['base_model_name']
    context_length = llm_config.get('context_length')
    use_unsloth = llm_config.get('use_unsloth', False)
    load_in_4bit = llm_config.get('load_in_4bit', True)
    
    print(f"  Loading LLM: {base_model_name}")
    model_loader = LLM(
        base_model_name,
        quantization_config_4bit,
        context_length=context_length,
        use_unsloth=use_unsloth,
        load_in_4bit=load_in_4bit
    )
    model = model_loader.get_model()
    tokenizer = model_loader.get_tokenizer()
    
    # Load LoRA adapters if available
    if lora_path and lora_path.exists():
        print(f"  Loading LoRA adapters from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    
    # Load prompt config
    with open(conf_test_dir / config.nn_gen_conf) as f:
        prompt_dict = json.load(f)
    
    # Create chat bot
    chat_bot = ChatBot(model, tokenizer, temperature=0.8, top_k=70, top_p=0.9)
    
    # Generate (without NNEval call)
    nn_gen(config, cycle, chat_bot, prompt_dict)
    
    release_memory()
    return synth_dir


def step2_evaluate_models(config: PipelineConfig, cycle: int, synth_dir: Path) -> CycleMetrics:
    """
    Step 2: Evaluate generated models on CIFAR-10.
    
    Uses NNEval.main() with pipeline_nneval_dir to avoid epoch_dir() issues.
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Evaluate Models on CIFAR-10 (Cycle {cycle})")
    print(f"{'='*60}")
    
    if config.dry_run:
        print(f"  [DRY RUN] Would evaluate models in: {synth_dir}")
        dummy_accs = [0.3 + 0.02 * cycle + 0.1 * i / 10 for i in range(10)]
        return CycleMetrics(
            cycle=cycle,
            accuracies=dummy_accs,
            num_generated=10,
            num_valid=10,
            num_evaluated=10
        )
    
    from ab.gpt import NNEval
    
    print(f"  Running NNEval on {synth_dir}...")
    NNEval.main(
        nn_name_prefix=f"cyc{cycle}",
        nn_train_epochs=config.nn_train_epochs,
        save_to_db=True,
        cycle=cycle,
        nn_alter_epochs=1,
        pipeline_nneval_dir=str(synth_dir)
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


def step3_deduplicate_code(config: PipelineConfig, cycle: int) -> Path:
    """
    Step 3: Deduplicate generated code using nn-dup.
    
    Runs the deduplication pipeline and ChatPrep to generate
    train.jsonl/dev.jsonl/test.jsonl for LLM fine-tuning.
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Deduplicate Code with nn-dup (Cycle {cycle})")
    print(f"{'='*60}")
    
    chat_data_dir = config.chat_data_dir(cycle)
    chat_data_dir.mkdir(parents=True, exist_ok=True)
    
    if config.dry_run or config.skip_nn_dup:
        print(f"  [{'DRY RUN' if config.dry_run else 'SKIP'}] Would run nn-dup pipeline")
        print(f"  Output dir: {chat_data_dir}")
        return chat_data_dir
    
    # Import nn-dup modules
    try:
        from ab.dup.preprocessing import curate_from_lemur
        from ab.chatprep import ChatPrepConfig
    except ImportError as e:
        print(f"  [WARN] nn-dup not available: {e}")
        print(f"  Falling back to using existing LEMUR data...")
        return chat_data_dir
    
    # Step 3a: Run deduplication
    dedup_output = config.cycle_dir(cycle) / 'dedup_output'
    dedup_output.mkdir(parents=True, exist_ok=True)
    
    print(f"  Running deduplication pipeline...")
    import logging
    logger = logging.getLogger('nn-dup')
    logger.setLevel(logging.INFO)
    
    curate_from_lemur(
        out_dir=dedup_output,
        includes=[f"cyc{cycle}"],  # Filter for this cycle's models
        prefer_order=[f"cyc{cycle}"],
        min_per_prefix=1,
        keep_per_family=5,
        logger=logger
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
            drop_unparseable=True
        )
        chat_config.run()
    else:
        print(f"  [WARN] No accepted code found, skipping ChatPrep")
    
    return chat_data_dir


def step4_finetune_llm(config: PipelineConfig, cycle: int, chat_data_dir: Path) -> Path:
    """
    Step 4: Fine-tune LLM with QLoRA using deduplicated data.
    
    Uses the chat_data from nn-dup to train the LLM.
    Saves LoRA adapters for the next cycle.
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: Fine-tune LLM with QLoRA (Cycle {cycle})")
    print(f"{'='*60}")
    
    lora_output = config.cycle_dir(cycle) / 'lora_checkpoint'
    lora_output.mkdir(parents=True, exist_ok=True)
    
    if config.dry_run:
        print(f"  [DRY RUN] Would fine-tune LLM with QLoRA")
        print(f"  Data dir: {chat_data_dir}")
        print(f"  Output: {lora_output}")
        return lora_output
    
    # Check if training data exists
    train_file = chat_data_dir / 'train.jsonl'
    if not train_file.exists():
        print(f"  [WARN] No training data found at {train_file}")
        print(f"  Skipping fine-tuning for this cycle")
        return lora_output
    
    # Import TuneNNGen
    from ab.gpt import TuneNNGen
    
    # Get previous LoRA checkpoint if available
    prev_lora = config.lora_path(cycle)
    
    print(f"  Training with QLoRA (r={config.r}, alpha={config.lora_alpha})")
    print(f"  Previous LoRA: {prev_lora}")
    print(f"  Data dir: {chat_data_dir}")
    
    TuneNNGen.main(
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        llm_conf=config.llm_conf,
        peft=str(prev_lora) if prev_lora else None,
        data_dir=str(chat_data_dir),
        nn_name_prefix=f"cyc{cycle}",
        skip_epoches=-1  # Don't skip any epochs
    )
    
    release_memory()
    return lora_output


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
# Main Pipeline
# =============================================================================

class CyclicTrainingPipeline:
    """
    Main pipeline class that orchestrates all training cycles.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics_history: List[CycleMetrics] = []
        
        # Create output directory
        config.output_base.mkdir(parents=True, exist_ok=True)
    
    def run_cycle(self, cycle: int) -> CycleMetrics:
        """Run a single training cycle."""
        print(f"\n{'#'*70}")
        print(f"# CYCLE {cycle}/{self.config.num_cycles}")
        print(f"{'#'*70}")
        
        # Get LoRA path from previous cycle
        lora_path = self.config.lora_path(cycle)
        
        # Step 1: Generate vision models
        synth_dir = step1_generate_vision_models(self.config, cycle, lora_path)
        
        # Step 2: Evaluate models on CIFAR-10
        metrics = step2_evaluate_models(self.config, cycle, synth_dir)
        
        # Step 3: Deduplicate code with nn-dup
        chat_data_dir = step3_deduplicate_code(self.config, cycle)
        
        # Step 4: Fine-tune LLM with QLoRA
        step4_finetune_llm(self.config, cycle, chat_data_dir)
        
        return metrics
    
    def run(self):
        """Run all training cycles."""
        print(f"\n{'='*70}")
        print("CYCLIC TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Number of cycles: {self.config.num_cycles}")
        print(f"LLM config: {self.config.llm_conf}")
        print(f"Output directory: {self.config.output_base}")
        print(f"Dry run: {self.config.dry_run}")
        
        for cycle in range(self.config.num_cycles):
            try:
                metrics = self.run_cycle(cycle)
                self.metrics_history.append(metrics)
                
                # Print running summary
                print(f"\n[Cycle {cycle}] Best={metrics.best:.4f}, Avg={metrics.average:.4f}, Med={metrics.median:.4f}")
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                print(f"\n[ERROR] Cycle {cycle} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Add zero metrics for failed cycle
                self.metrics_history.append(CycleMetrics(
                    cycle=cycle,
                    accuracies=[],
                    num_generated=0,
                    num_valid=0,
                    num_evaluated=0
                ))
        
        # Final results
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
        skip_nn_dup=args.skip_nn_dup
    )
    
    if args.output:
        config.output_base = Path(args.output)
    
    # Run pipeline
    pipeline = CyclicTrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
