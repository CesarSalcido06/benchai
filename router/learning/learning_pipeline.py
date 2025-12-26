"""
Learning Pipeline - Orchestrates LoRA Fine-Tuning for Self-Improvement

This module manages the complete fine-tuning lifecycle:
1. Data collection from experience replay and interaction logs
2. Data cleaning and quality filtering
3. Training orchestration with Unsloth
4. Adapter management (multiple specialized LoRAs)
5. Evaluation and rollback

Based on research:
- Unsloth for 2.5x faster training, 90% less VRAM
- QLoRA for efficient quantized training
- Progressive adapters to prevent catastrophic forgetting
"""

import asyncio
import aiosqlite
import json
import os
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .experience_replay import ExperienceReplayManager, TaskDomain
from .interaction_logger import InteractionLogger, InteractionType


class AdapterType(str, Enum):
    """Types of LoRA adapters for different specializations."""
    RESEARCH = "research"
    ORCHESTRATION = "orchestration"
    CODING = "coding"
    GENERAL = "general"


class TrainingStatus(str, Enum):
    """Status of a training run."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    adapter_type: AdapterType
    base_model: str
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    num_epochs: int = 3
    warmup_steps: int = 10
    max_seq_length: int = 2048
    use_4bit: bool = True  # QLoRA
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class TrainingRun:
    """A single training run."""
    id: Optional[int]
    adapter_type: AdapterType
    config: TrainingConfig
    status: TrainingStatus
    training_examples: int
    validation_examples: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metrics: Dict[str, Any]
    adapter_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime


class LearningPipeline:
    """
    Orchestrates the continuous learning pipeline.

    Workflow:
    1. Collect training data from experience replay + interactions
    2. Clean and format data for instruction tuning
    3. Train LoRA adapter using Unsloth
    4. Evaluate on held-out set
    5. Deploy if better, rollback if worse
    """

    def __init__(
        self,
        db_path: Path,
        models_dir: Path,
        adapters_dir: Path,
        experience_manager: ExperienceReplayManager,
        interaction_logger: InteractionLogger
    ):
        self.db_path = db_path
        self.models_dir = models_dir
        self.adapters_dir = adapters_dir
        self.experience_manager = experience_manager
        self.interaction_logger = interaction_logger

        self._initialized = False
        self._training_lock = asyncio.Lock()
        self._current_training: Optional[TrainingRun] = None

        # Training thresholds
        self.min_training_examples = 50
        self.min_quality_score = 0.7
        self.validation_split = 0.1
        self.eval_improvement_threshold = 0.02  # 2% improvement required

    async def initialize(self):
        """Initialize the learning pipeline database."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Training runs table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_type TEXT NOT NULL,
                    config JSON NOT NULL,
                    status TEXT DEFAULT 'pending',
                    training_examples INTEGER DEFAULT 0,
                    validation_examples INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metrics JSON DEFAULT '{}',
                    adapter_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Active adapters table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS active_adapters (
                    adapter_type TEXT PRIMARY KEY,
                    training_run_id INTEGER,
                    adapter_path TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    activated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    eval_score REAL,
                    FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
                )
            ''')

            # Training data tracking
            await db.execute('''
                CREATE TABLE IF NOT EXISTS training_data_used (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_run_id INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
                )
            ''')

            await db.commit()

        self._initialized = True
        print("[LEARNING] Learning pipeline initialized")

    async def collect_training_data(
        self,
        adapter_type: AdapterType,
        min_examples: Optional[int] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Collect and prepare training data from multiple sources.

        Returns:
            Tuple of (training_data, validation_data)
        """
        await self.initialize()

        min_examples = min_examples or self.min_training_examples
        all_examples = []

        # Map adapter type to domains
        domain_mapping = {
            AdapterType.RESEARCH: [TaskDomain.RESEARCH],
            AdapterType.ORCHESTRATION: [TaskDomain.ORCHESTRATION, TaskDomain.SYSTEM],
            AdapterType.CODING: [TaskDomain.CODING],
            AdapterType.GENERAL: [TaskDomain.GENERAL, TaskDomain.CREATIVE],
        }
        domains = domain_mapping.get(adapter_type, [])

        # 1. Get training data from experience replay
        for domain in domains:
            training_data = await self.experience_manager.get_training_data(
                domain=domain.value,
                min_quality=self.min_quality_score,
                unused_only=True,
                limit=min_examples
            )
            for item in training_data:
                all_examples.append({
                    "instruction": item['instruction'],
                    "input": item.get('input', ''),
                    "output": item['output'],
                    "source": "experience",
                    "source_id": item['experience_id'],
                    "quality": item['quality_score']
                })

        # 2. Get high-quality interactions with positive feedback
        interaction_type_mapping = {
            AdapterType.RESEARCH: InteractionType.RAG_QUERY,
            AdapterType.ORCHESTRATION: InteractionType.AGENT_CALL,
            AdapterType.CODING: InteractionType.TOOL_USE,
            AdapterType.GENERAL: InteractionType.CHAT,
        }
        int_type = interaction_type_mapping.get(adapter_type)

        high_quality = await self.interaction_logger.get_high_quality_interactions(
            min_feedback=4,
            interaction_type=int_type,
            limit=min_examples
        )

        for interaction in high_quality:
            # Convert to instruction format
            request = interaction['request']
            response = interaction['response']

            if 'messages' in request and response.get('choices'):
                # Chat format
                user_msg = next(
                    (m['content'] for m in request['messages'] if m.get('role') == 'user'),
                    None
                )
                assistant_msg = response['choices'][0].get('message', {}).get('content', '')

                if user_msg and assistant_msg:
                    all_examples.append({
                        "instruction": user_msg,
                        "input": "",
                        "output": assistant_msg,
                        "source": "interaction",
                        "source_id": interaction['id'],
                        "quality": interaction['user_feedback'] / 5.0
                    })

        # 3. Sort by quality and deduplicate
        all_examples.sort(key=lambda x: x['quality'], reverse=True)

        # Simple deduplication by instruction similarity
        seen_instructions = set()
        unique_examples = []
        for ex in all_examples:
            instruction_key = ex['instruction'][:100].lower()
            if instruction_key not in seen_instructions:
                seen_instructions.add(instruction_key)
                unique_examples.append(ex)

        # Split into train/validation
        n_val = max(1, int(len(unique_examples) * self.validation_split))
        validation_data = unique_examples[:n_val]
        training_data = unique_examples[n_val:]

        print(f"[LEARNING] Collected {len(training_data)} training, {len(validation_data)} validation examples")

        return training_data, validation_data

    def _format_for_training(self, examples: List[Dict], output_path: Path):
        """Format examples in Alpaca/instruction format for Unsloth."""
        formatted = []
        for ex in examples:
            formatted.append({
                "instruction": ex['instruction'],
                "input": ex.get('input', ''),
                "output": ex['output']
            })

        with open(output_path, 'w') as f:
            json.dump(formatted, f, indent=2)

        return output_path

    async def create_training_run(
        self,
        adapter_type: AdapterType,
        config: Optional[TrainingConfig] = None
    ) -> int:
        """Create a new training run."""
        await self.initialize()

        if config is None:
            # Default config based on adapter type
            base_models = {
                AdapterType.RESEARCH: "mistralai/Mistral-7B-Instruct-v0.3",
                AdapterType.ORCHESTRATION: "mistralai/Mistral-7B-Instruct-v0.3",
                AdapterType.CODING: "Qwen/Qwen2.5-Coder-7B-Instruct",
                AdapterType.GENERAL: "microsoft/Phi-3-mini-4k-instruct",
            }
            config = TrainingConfig(
                adapter_type=adapter_type,
                base_model=base_models.get(adapter_type, "mistralai/Mistral-7B-Instruct-v0.3")
            )

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                '''INSERT INTO training_runs (adapter_type, config, status)
                   VALUES (?, ?, 'pending')''',
                (adapter_type.value, json.dumps({
                    "base_model": config.base_model,
                    "lora_r": config.lora_r,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "gradient_accumulation": config.gradient_accumulation,
                    "num_epochs": config.num_epochs,
                    "warmup_steps": config.warmup_steps,
                    "max_seq_length": config.max_seq_length,
                    "use_4bit": config.use_4bit,
                    "target_modules": config.target_modules
                }))
            )
            run_id = cursor.lastrowid
            await db.commit()

        return run_id

    async def run_training(self, run_id: int) -> Dict:
        """
        Execute a training run.

        This generates a training script and runs it with Unsloth.
        """
        await self.initialize()

        async with self._training_lock:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                cursor = await db.execute(
                    "SELECT * FROM training_runs WHERE id = ?",
                    (run_id,)
                )
                run = await cursor.fetchone()

                if not run:
                    raise ValueError(f"Training run {run_id} not found")

                if run['status'] not in ['pending', 'failed']:
                    raise ValueError(f"Training run {run_id} is not in pending state")

                adapter_type = AdapterType(run['adapter_type'])
                config = json.loads(run['config'])

                # Update status to preparing
                await db.execute(
                    "UPDATE training_runs SET status = 'preparing', started_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (run_id,)
                )
                await db.commit()

            try:
                # Collect training data
                training_data, validation_data = await self.collect_training_data(adapter_type)

                if len(training_data) < self.min_training_examples:
                    raise ValueError(f"Insufficient training data: {len(training_data)} < {self.min_training_examples}")

                # Create training directory
                train_dir = self.adapters_dir / f"run_{run_id}"
                train_dir.mkdir(parents=True, exist_ok=True)

                # Save training data
                train_file = train_dir / "train.json"
                val_file = train_dir / "val.json"
                self._format_for_training(training_data, train_file)
                self._format_for_training(validation_data, val_file)

                # Generate training script
                script_path = train_dir / "train.py"
                self._generate_training_script(
                    script_path=script_path,
                    config=config,
                    train_file=train_file,
                    val_file=val_file,
                    output_dir=train_dir / "output"
                )

                # Update status and counts
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        '''UPDATE training_runs SET
                           status = 'training',
                           training_examples = ?,
                           validation_examples = ?
                           WHERE id = ?''',
                        (len(training_data), len(validation_data), run_id)
                    )
                    await db.commit()

                # Run training
                result = subprocess.run(
                    ["python3", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Training failed: {result.stderr}")

                # Load metrics from output
                metrics_file = train_dir / "output" / "training_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                else:
                    metrics = {"status": "completed", "log": result.stdout[-1000:]}

                adapter_path = str(train_dir / "output" / "adapter")

                # Update with success
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        '''UPDATE training_runs SET
                           status = 'completed',
                           completed_at = CURRENT_TIMESTAMP,
                           metrics = ?,
                           adapter_path = ?
                           WHERE id = ?''',
                        (json.dumps(metrics), adapter_path, run_id)
                    )

                    # Track which data was used
                    for ex in training_data:
                        await db.execute(
                            '''INSERT INTO training_data_used (training_run_id, source_type, source_id)
                               VALUES (?, ?, ?)''',
                            (run_id, ex['source'], ex['source_id'])
                        )

                    await db.commit()

                # Mark experience training data as used
                experience_ids = [ex['source_id'] for ex in training_data if ex['source'] == 'experience']
                await self.experience_manager.mark_training_used(experience_ids)

                return {
                    "status": "completed",
                    "run_id": run_id,
                    "adapter_path": adapter_path,
                    "metrics": metrics,
                    "training_examples": len(training_data),
                    "validation_examples": len(validation_data)
                }

            except Exception as e:
                # Update with failure
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        '''UPDATE training_runs SET
                           status = 'failed',
                           completed_at = CURRENT_TIMESTAMP,
                           error_message = ?
                           WHERE id = ?''',
                        (str(e), run_id)
                    )
                    await db.commit()

                raise

    def _generate_training_script(
        self,
        script_path: Path,
        config: Dict,
        train_file: Path,
        val_file: Path,
        output_dir: Path
    ):
        """Generate a Python training script using Unsloth."""
        script = f'''#!/usr/bin/env python3
"""Auto-generated training script for BenchAI LoRA fine-tuning."""

import json
import torch
from pathlib import Path

# Check for Unsloth availability
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available, using transformers fallback")

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
BASE_MODEL = "{config['base_model']}"
LORA_R = {config['lora_r']}
LORA_ALPHA = {config['lora_alpha']}
LORA_DROPOUT = {config['lora_dropout']}
LEARNING_RATE = {config['learning_rate']}
BATCH_SIZE = {config['batch_size']}
GRADIENT_ACCUMULATION = {config['gradient_accumulation']}
NUM_EPOCHS = {config['num_epochs']}
WARMUP_STEPS = {config['warmup_steps']}
MAX_SEQ_LENGTH = {config['max_seq_length']}
USE_4BIT = {config['use_4bit']}
TARGET_MODULES = {config['target_modules']}

TRAIN_FILE = "{train_file}"
VAL_FILE = "{val_file}"
OUTPUT_DIR = "{output_dir}"

def format_prompt(example):
    """Format example in Alpaca style."""
    if example.get("input"):
        return f"""### Instruction:
{{example["instruction"]}}

### Input:
{{example["input"]}}

### Response:
{{example["output"]}}"""
    else:
        return f"""### Instruction:
{{example["instruction"]}}

### Response:
{{example["output"]}}"""

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")

    print(f"Training examples: {{len(train_dataset)}}")
    print(f"Validation examples: {{len(val_dataset)}}")

    if UNSLOTH_AVAILABLE:
        # Use Unsloth for faster training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # Auto-detect
            load_in_4bit=USE_4BIT,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        # Fallback to transformers + PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_4bit=USE_4BIT,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format datasets
    train_dataset = train_dataset.map(lambda x: {{"text": format_prompt(x)}})
    val_dataset = val_dataset.map(lambda x: {{"text": format_prompt(x)}})

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported() if UNSLOTH_AVAILABLE else True,
        bf16=is_bfloat16_supported() if UNSLOTH_AVAILABLE else False,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        optim="adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save adapter
    adapter_path = Path(OUTPUT_DIR) / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Save metrics
    metrics = {{
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
    }}

    # Evaluate
    eval_results = trainer.evaluate()
    metrics["eval_loss"] = eval_results.get("eval_loss")
    metrics["eval_runtime"] = eval_results.get("eval_runtime")

    with open(Path(OUTPUT_DIR) / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete! Adapter saved to {{adapter_path}}")
    print(f"Metrics: {{metrics}}")

if __name__ == "__main__":
    main()
'''

        with open(script_path, 'w') as f:
            f.write(script)

        os.chmod(script_path, 0o755)

    async def activate_adapter(self, run_id: int) -> bool:
        """Activate a trained adapter for use."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                "SELECT * FROM training_runs WHERE id = ? AND status = 'completed'",
                (run_id,)
            )
            run = await cursor.fetchone()

            if not run:
                raise ValueError(f"Completed training run {run_id} not found")

            adapter_type = run['adapter_type']
            config = json.loads(run['config'])
            metrics = json.loads(run['metrics'])

            await db.execute(
                '''INSERT OR REPLACE INTO active_adapters
                   (adapter_type, training_run_id, adapter_path, base_model, eval_score)
                   VALUES (?, ?, ?, ?, ?)''',
                (adapter_type, run_id, run['adapter_path'], config['base_model'],
                 metrics.get('eval_loss'))
            )
            await db.commit()

        print(f"[LEARNING] Activated adapter: {adapter_type} from run {run_id}")
        return True

    async def get_active_adapters(self) -> Dict[str, Dict]:
        """Get all currently active adapters."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("SELECT * FROM active_adapters")
            adapters = {}
            async for row in cursor:
                adapters[row['adapter_type']] = {
                    "training_run_id": row['training_run_id'],
                    "adapter_path": row['adapter_path'],
                    "base_model": row['base_model'],
                    "activated_at": row['activated_at'],
                    "eval_score": row['eval_score']
                }

            return adapters

    async def get_training_runs(
        self,
        adapter_type: Optional[AdapterType] = None,
        status: Optional[TrainingStatus] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get training run history."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            conditions = []
            params = []

            if adapter_type:
                conditions.append("adapter_type = ?")
                params.append(adapter_type.value)

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)

            cursor = await db.execute(f'''
                SELECT * FROM training_runs
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            ''', params)

            runs = []
            async for row in cursor:
                run = dict(row)
                run['config'] = json.loads(run['config'])
                run['metrics'] = json.loads(run['metrics']) if run['metrics'] else {}
                runs.append(run)

            return runs

    async def should_trigger_training(self, adapter_type: AdapterType) -> Tuple[bool, str]:
        """
        Check if training should be triggered for an adapter type.

        Returns:
            Tuple of (should_train, reason)
        """
        await self.initialize()

        # Check for pending training data
        training_data, _ = await self.collect_training_data(adapter_type)

        if len(training_data) < self.min_training_examples:
            return False, f"Insufficient data: {len(training_data)}/{self.min_training_examples}"

        # Check last training time
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                '''SELECT completed_at FROM training_runs
                   WHERE adapter_type = ? AND status = 'completed'
                   ORDER BY completed_at DESC LIMIT 1''',
                (adapter_type.value,)
            )
            row = await cursor.fetchone()

            if row and row[0]:
                last_training = datetime.fromisoformat(row[0])
                days_since = (datetime.now() - last_training).days

                if days_since < 7:
                    return False, f"Recent training {days_since} days ago"

        return True, f"Ready with {len(training_data)} examples"

    async def get_stats(self) -> Dict:
        """Get learning pipeline statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            stats = {"runs_by_status": {}, "runs_by_type": {}, "active_adapters": 0}

            # By status
            cursor = await db.execute(
                "SELECT status, COUNT(*) FROM training_runs GROUP BY status"
            )
            async for row in cursor:
                stats["runs_by_status"][row[0]] = row[1]

            # By type
            cursor = await db.execute(
                "SELECT adapter_type, COUNT(*) FROM training_runs GROUP BY adapter_type"
            )
            async for row in cursor:
                stats["runs_by_type"][row[0]] = row[1]

            # Active adapters
            cursor = await db.execute("SELECT COUNT(*) FROM active_adapters")
            stats["active_adapters"] = (await cursor.fetchone())[0]

            # Total training examples used
            cursor = await db.execute("SELECT COUNT(*) FROM training_data_used")
            stats["total_examples_used"] = (await cursor.fetchone())[0]

            return stats
