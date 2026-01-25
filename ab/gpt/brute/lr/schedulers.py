import os
import shutil
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

# Try to import transformers for HF schedulers (optional)
try:
    from transformers import (
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_inverse_sqrt_schedule,
        get_cosine_with_min_lr_schedule_with_warmup,
    )
    HF_SCHEDULERS_AVAILABLE = True
except ImportError:
    HF_SCHEDULERS_AVAILABLE = False
    print("⚠️ Transformers not available. HuggingFace schedulers will be skipped.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Target the directory NNEval is scanning
OUTPUT_DIR = "out/nngpt/llm/epoch/A0/synth_nn"

# Use "A0_" prefix to match the parent folder "epoch/A0".
BASE_MODEL_NAME = "A0_"

# -----------------------------------------------------------------------------
# 0. Setup & Cleanup
# -----------------------------------------------------------------------------
print(f"--- Setup Phase ---")

# 1. Ensure the full path exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Created/Verified model directory: {OUTPUT_DIR}")

# 2. Recursive __init__.py creation
path_parts = pathlib.Path(OUTPUT_DIR).parts
current_path = ""
for part in path_parts:
    current_path = os.path.join(current_path, part) if current_path else part
    init_file = os.path.join(current_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Package marker for NNEval\n")
        print(f"    + Created missing package marker: {init_file}")

# 3. Clean up OLD generated files
print("--- Cleaning up previous files ---")
pycache_path = os.path.join(OUTPUT_DIR, "__pycache__")
if os.path.exists(pycache_path):
    shutil.rmtree(pycache_path)
    print(f"✅ Removed directory confusing NNEval: {pycache_path}")

# 4. Auto-delete the database to ensure a fresh scan
db_path = "db/ab.nn.db"
if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"✅ Deleted old database ({db_path}) to force a fresh scan.")
    except Exception as e:
        print(f"⚠️ Could not delete database: {e}")

# -----------------------------------------------------------------------------
# 1. The Model Template
# -----------------------------------------------------------------------------
model_template = """
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler

# Supported hyperparameters for this model
def supported_hyperparameters():
    return {supported_hyperparameters_set}

# -------------------------------------------------------------------------
# Model Definition: Net (ResNet18 adapted for CIFAR-10)
# -------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, in_shape=(1, 3, 32, 32), out_shape=(10,), prm=None, device='cpu'):
        '''
        Initialize ResNet18 model.
        Parameters match NNEval's expected signature.
        '''
        super(Net, self).__init__()
        
        if prm is None:
            prm = {{'lr': 0.1, 'momentum': 0.9, 'dropout': 0.2, 'epoch_max': 90}}
        
        self.device = device
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        # Initialize standard ResNet18
        self.model = models.resnet18(weights=None)
        
        # Modify the final layer to include Dropout and set correct output classes
        num_classes = out_shape[0] if isinstance(out_shape, tuple) else out_shape
        self.model.fc = nn.Sequential(
            nn.Dropout(p=prm.get('dropout', 0.2)),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        # Training components
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)
    
    def train_setup(self, prm):
        '''
        Initialize training components (criterion, optimizer, scheduler).
        Required by NNEval for training evaluation.
        '''
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Initialize Optimizer
        self.optimizer = optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.1), 
            momentum=prm.get('momentum', 0.9),
            weight_decay=1e-4
        )
        
        # Create local variable reference for scheduler creation
        optimizer = self.optimizer
        
        # --- SCHEDULER LOGIC START ---
        {scheduler_logic}
        # --- SCHEDULER LOGIC END ---
        
        self.scheduler = scheduler
    
    def learn(self, train_data):
        '''
        Training loop. Process one batch or epoch of training data.
        '''
        self.train()
        for (inputs, labels) in train_data:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                print(f"[WARN] Skipping batch due to non-finite loss: {{loss.item()}}")
                continue
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3.0)
            self.optimizer.step()
        
        # Step scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()

# -------------------------------------------------------------------------
# Hyperparameters & Scheduler Configuration
# Strategy: {strategy_name}
# -------------------------------------------------------------------------
def get_model_and_optimizer(in_shape=(1, 3, 32, 32), out_shape=(10,), max_epoch=90, learning_rate=0.1, dropout=0.2, momentum=0.9, device='cpu', **kwargs):
    '''
    Factory function required by NNEval.
    '''
    # Prepare hyperparameters - include all scheduler-specific parameters
    prm = {{
        'lr': learning_rate,
        'momentum': momentum,
        'dropout': dropout,
        'epoch_max': max_epoch
    }}
    
    # Add any additional scheduler-specific parameters from kwargs
    prm.update(kwargs)
    
    # Initialize model with the parameters
    model = Net(in_shape=in_shape, out_shape=out_shape, prm=prm, device=device)
    
    # Setup training (this will also initialize optimizer and scheduler)
    model.train_setup(prm)
    
    return model, model.optimizer, model.scheduler

# -------------------------------------------------------------------------
# Metadata for LLM Analysis
# -------------------------------------------------------------------------
# description: ResNet18 model using {strategy_name}
# category: Computer Vision
# scheduler_type: {scheduler_type}
"""

# ============================================================================
# PyTorch Standard Schedulers (A) - with prm-based hyperparameters
# ============================================================================
pytorch_strategies = [
    # Step-based decay strategies
    {
        "name": "StepLR_default",
        "type": "Step-based",
        "hyperparams": {"step_size", "gamma"},
        "code": "scheduler = lr_scheduler.StepLR(optimizer, step_size=prm.get('step_size', 30), gamma=prm.get('gamma', 0.1))"
    },
    {
        "name": "MultiStepLR_milestones",
        "type": "Step-based",
        "hyperparams": {"milestone0", "milestone1", "milestone2", "gamma"},
        "code": "scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=sorted([int(prm['epoch_max'] * prm.get('milestone0', 0.3)), int(prm['epoch_max'] * prm.get('milestone1', 0.6)), int(prm['epoch_max'] * prm.get('milestone2', 0.8))]), gamma=prm.get('gamma', 0.1))"
    },
    {
        "name": "ExponentialLR_default",
        "type": "Step-based",
        "hyperparams": {"gamma"},
        "code": "scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=prm.get('gamma', 0.95))"
    },
    
    # Polynomial decay
    {
        "name": "PolynomialLR_quadratic",
        "type": "Polynomial",
        "hyperparams": {"power"},
        "code": "scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=prm['epoch_max'], power=prm.get('power', 1.0))"
    },
    {
        "name": "PolynomialLR_cubic",
        "type": "Polynomial",
        "hyperparams": {"power"},
        "code": "scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=prm['epoch_max'], power=prm.get('power', 2.0))"
    },
    
    # Cosine annealing strategies
    {
        "name": "CosineAnnealingLR_default",
        "type": "Cosine",
        "hyperparams": {"eta_min"},
        "code": "scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm['epoch_max'], eta_min=prm.get('eta_min', 0.0))"
    },
    {
        "name": "CosineAnnealingLR_with_min_lr",
        "type": "Cosine",
        "hyperparams": {"min_lr"},
        "code": "scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=prm['epoch_max'], eta_min=prm.get('lr', 0.1) * prm.get('min_lr', 0.01))"
    },
    {
        "name": "CosineAnnealingWarmRestarts_T0_small",
        "type": "Cosine",
        "hyperparams": {"T_0", "T_mult", "eta_min"},
        "code": "scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(prm.get('T_0', 10)), T_mult=prm.get('T_mult', 2), eta_min=prm.get('eta_min', 0.0))"
    },
    {
        "name": "CosineAnnealingWarmRestarts_T0_large",
        "type": "Cosine",
        "hyperparams": {"T_0", "T_mult", "eta_min"},
        "code": "scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(prm.get('T_0', 20)), T_mult=prm.get('T_mult', 1), eta_min=prm.get('eta_min', 0.0))"
    },
    
    # Cyclic LR with different modes
    {
        "name": "CyclicLR_triangular",
        "type": "Cyclic",
        "hyperparams": {"lr", "min_lr", "lr_step"},
        "code": "scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=int(prm['epoch_max'] * prm.get('lr_step', 0.1)), mode='triangular')"
    },
    {
        "name": "CyclicLR_triangular2",
        "type": "Cyclic",
        "hyperparams": {"lr", "min_lr", "lr_step"},
        "code": "scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=int(prm['epoch_max'] * prm.get('lr_step', 0.1)), mode='triangular2')"
    },
    {
        "name": "CyclicLR_exp_range",
        "type": "Cyclic",
        "hyperparams": {"lr", "min_lr", "lr_step", "cycle_momentum"},
        "code": "scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=prm.get('lr', 0.1) * prm.get('min_lr', 0.1), max_lr=prm.get('lr', 0.1), step_size_up=int(prm['epoch_max'] * prm.get('lr_step', 0.1)), mode='exp_range', gamma=prm.get('cycle_momentum', 0.85))"
    },
    
    # One cycle policy
    {
        "name": "OneCycleLR_default",
        "type": "Cyclic",
        "hyperparams": {"lr", "min_lr"},
        "code": "scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=prm.get('lr', 0.1), total_steps=prm['epoch_max'], pct_start=prm.get('pct_start', 0.3), anneal_strategy='cos', div_factor=prm.get('div_factor', 25.0))"
    },
    {
        "name": "OneCycleLR_aggressive",
        "type": "Cyclic",
        "hyperparams": {"lr", "div_factor", "pct_start"},
        "code": "scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=prm.get('lr', 0.1), total_steps=prm['epoch_max'], pct_start=prm.get('pct_start', 0.1), anneal_strategy='linear', div_factor=prm.get('div_factor', 10.0))"
    },
    
    # Warmup-based strategies
    {
        "name": "LinearLR_warmup",
        "type": "Warmup",
        "hyperparams": {"warmup_epochs", "target_lr"},
        "code": "scheduler = lr_scheduler.LinearLR(optimizer, start_factor=prm.get('start_factor', 0.1), total_iters=int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1)))"
    },
    {
        "name": "ConstantLR_warmup",
        "type": "Warmup",
        "hyperparams": {"warmup_epochs", "factor"},
        "code": "scheduler = lr_scheduler.ConstantLR(optimizer, factor=prm.get('factor', 0.5), total_iters=int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1)))"
    },
    
    # LambdaLR for custom schedules
    {
        "name": "LambdaLR_linear_decay",
        "type": "Lambda",
        "hyperparams": {"power"},
        "code": "scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.1, (1.0 - epoch / prm['epoch_max'])))"
    },
    {
        "name": "LambdaLR_power_decay",
        "type": "Lambda",
        "hyperparams": {"power"},
        "code": "scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1.0 - epoch / prm['epoch_max']) ** prm.get('power', 2.0))"
    },
    
    # MultiplicativeLR
    {
        "name": "MultiplicativeLR_exponential",
        "type": "Multiplicative",
        "hyperparams": {"gamma"},
        "code": "scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: prm.get('gamma', 0.95))"
    },
]

# ============================================================================
# HuggingFace Transformers Schedulers (B) - if available
# ============================================================================
hf_strategies = []

if HF_SCHEDULERS_AVAILABLE:
    hf_strategies = [
        {
            "name": "HF_linear_warmup",
            "type": "Warmup",
            "hyperparams": {"warmup_epochs"},
            "code": "from transformers import get_linear_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=prm['epoch_max'])"
        },
        {
            "name": "HF_cosine_warmup",
            "type": "Cosine",
            "hyperparams": {"warmup_epochs", "eta_min"},
            "code": "from transformers import get_cosine_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=prm['epoch_max'])"
        },
        {
            "name": "HF_cosine_hard_restarts",
            "type": "Cosine",
            "hyperparams": {"warmup_epochs", "num_cycles"},
            "code": "from transformers import get_cosine_with_hard_restarts_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=prm['epoch_max'], num_cycles=prm.get('num_cycles', 1))"
        },
        {
            "name": "HF_polynomial_warmup",
            "type": "Polynomial",
            "hyperparams": {"warmup_epochs", "power"},
            "code": "from transformers import get_polynomial_decay_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=prm['epoch_max'], power=prm.get('power', 1.0))"
        },
        {
            "name": "HF_constant_warmup",
            "type": "Warmup",
            "hyperparams": {"warmup_epochs"},
            "code": "from transformers import get_constant_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)"
        },
        {
            "name": "HF_constant",
            "type": "Constant",
            "hyperparams": {},
            "code": "from transformers import get_constant_schedule\nscheduler = get_constant_schedule(optimizer)"
        },
        {
            "name": "HF_inverse_sqrt_warmup",
            "type": "Inverse Sqrt",
            "hyperparams": {"warmup_epochs"},
            "code": "from transformers import get_inverse_sqrt_schedule\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_inverse_sqrt_schedule(optimizer, timescale=max(1, int(prm['epoch_max'] / 10)))"
        },
        {
            "name": "HF_cosine_min_lr_warmup",
            "type": "Cosine",
            "hyperparams": {"warmup_epochs", "min_lr"},
            "code": "from transformers import get_cosine_with_min_lr_schedule_with_warmup\nwarmup_steps = int(prm['epoch_max'] * prm.get('warmup_epochs', 0.1))\nscheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=prm['epoch_max'], min_lr_ratio=prm.get('min_lr', 0.01))"
        },
    ]

# Combine all strategies
strategies = pytorch_strategies + hf_strategies

# ============================================================================
# 3. Execution: Generate the Files
# ============================================================================
print(f"\n--- Generation Phase ---")
print(f"Generating {len(strategies)} scheduler variants...")

generated_files = []
module_names = []

for i, strategy in enumerate(strategies):
    # Naming: A0_001, A0_002, etc.
    module_name = f"{BASE_MODEL_NAME}{i+1:03d}"
    
    # Create a subdirectory for each model (NNEval expects this structure)
    model_dir = os.path.join(OUTPUT_DIR, module_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # File goes inside the model directory as 'new_nn.py'
    filepath = os.path.join(model_dir, "new_nn.py")
    
    try:
        # Convert hyperparams set to string representation for supported_hyperparameters()
        hyperparams_set = strategy.get('hyperparams', set())
        # Add base hyperparams that are always needed
        base_hyperparams = {'lr', 'momentum', 'dropout'}
        all_hyperparams = base_hyperparams | hyperparams_set
        hyperparams_str = repr(sorted(all_hyperparams))
        
        # Fill the template
        file_content = model_template.format(
            strategy_name=strategy['name'],
            scheduler_type=strategy['type'],
            scheduler_logic=strategy['code'].strip(),
            supported_hyperparameters_set=hyperparams_str
        )
        
        with open(filepath, "w") as f:
            f.write(file_content)
            
        print(f"✅ Generated: {filepath}")
        generated_files.append(filepath)
        module_names.append(module_name)
    except Exception as e:
        print(f"❌ Error generating {filepath}: {e}")

# ============================================================================
# 4. Finalize __init__.py in synth_nn
# ============================================================================
init_path = os.path.join(OUTPUT_DIR, "__init__.py")

with open(init_path, "w") as f:
    f.write("# Package marker for NNEval\n")
    f.write("# Models are organized in subdirectories, each with new_nn.py\n")

print(f"✅ Created {init_path} as package marker.")

# ============================================================================
# 5. Final Cleanup and Verification
# ============================================================================
print("\n--- Final Cleanup and Verification ---")

pycache_path = os.path.join(OUTPUT_DIR, "__pycache__")
if os.path.exists(pycache_path):
    shutil.rmtree(pycache_path)
    print(f"✅ Rerunning cleanup: Removed directory {pycache_path}")

if generated_files and os.path.exists(generated_files[0]):
    print(f"✅ File 1 exists: {generated_files[0]}")
    print(f"✅ Total files generated: {len(generated_files)}")
elif generated_files:
    print(f"❌ Error: File 1 missing.")
else:
    print(f"❌ Error: No files were generated.")

print("\n=======================================================")
print("✅ PREPARATION COMPLETE")
print(f"   Generated {len(generated_files)} scheduler variants")
print("1. Run this script: python generate_lr_schedulers.py")
print("2. THEN run NNEval: python -m ab.gpt.NNEval")
print("======================================================="
