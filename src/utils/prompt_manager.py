"""
Prompt Manager - Prompt Templates and Variations

Manages prompts for different tasks and enables prompt ablation studies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """A single prompt template with metadata."""
    name: str
    system_prompt: str
    user_prompt: str
    task: str  # "rrg", "vqa", "grounding"
    version: str = "v1"
    description: str = ""
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    
    def format(self, **kwargs) -> Dict[str, str]:
        """Format the prompt with provided variables."""
        return {
            "system": self.system_prompt.format(**kwargs) if kwargs else self.system_prompt,
            "user": self.user_prompt.format(**kwargs) if kwargs else self.user_prompt,
        }


class PromptManager:
    """
    Manages prompt templates for different models and tasks.
    
    Supports:
    - Multiple prompt versions for A/B testing
    - Task-specific prompts (RRG, VQA, Grounding)
    - Model-specific prompt adaptations
    - Prompt ablation studies
    """
    
    # Default prompts for Report Generation
    DEFAULT_RRG_PROMPTS = {
        "baseline": PromptTemplate(
            name="baseline",
            task="rrg",
            version="v1",
            description="Minimal baseline prompt",
            system_prompt="You are a radiologist.",
            user_prompt="Generate a radiology report for this chest X-ray.",
        ),
        "detailed": PromptTemplate(
            name="detailed",
            task="rrg",
            version="v1",
            description="Detailed prompt with structure",
            system_prompt="""You are an expert radiologist with 20 years of experience. 
Your task is to analyze chest X-ray images and generate comprehensive radiology reports.""",
            user_prompt="""Analyze this chest X-ray and generate a detailed radiology report.

Your report MUST include:
1. FINDINGS: Systematic description of all observations including:
   - Heart size and cardiac silhouette
   - Lung fields (right and left)
   - Pleural spaces
   - Mediastinum
   - Bones and soft tissues

2. IMPRESSION: Concise clinical interpretation summarizing key findings.

Be specific about locations (right/left, upper/middle/lower zones) and severity.""",
        ),
        "structured": PromptTemplate(
            name="structured",
            task="rrg",
            version="v1",
            description="Highly structured format",
            system_prompt="You are an expert radiologist generating structured reports.",
            user_prompt="""Generate a structured radiology report for this chest X-ray.

Use this exact format:
---
TECHNIQUE: [imaging technique]
COMPARISON: [prior studies if available, otherwise "None"]
FINDINGS:
- Lungs: [description]
- Heart: [description]
- Mediastinum: [description]
- Pleura: [description]
- Bones: [description]
IMPRESSION:
1. [primary finding]
2. [secondary finding if any]
---""",
        ),
        "turkish": PromptTemplate(
            name="turkish",
            task="rrg",
            version="v1",
            description="Turkish language prompt",
            system_prompt="Deneyimli bir radyolog olarak görev yapıyorsunuz.",
            user_prompt="""Bu göğüs röntgenini analiz edin ve aşağıdaki formatta bir radyoloji raporu oluşturun:

BULGULAR: Tüm gözlemlerin sistematik açıklaması
İZLENİM: Temel bulguların klinik yorumu

Lokasyonlar (sağ/sol, üst/orta/alt zonlar) ve şiddet hakkında spesifik olun.""",
        ),
        "chain_of_thought": PromptTemplate(
            name="chain_of_thought",
            task="rrg",
            version="v1",
            description="Chain-of-thought reasoning prompt",
            system_prompt="""You are an expert radiologist. Think step by step when analyzing images.""",
            user_prompt="""Analyze this chest X-ray step by step:

Step 1: First, assess the technical quality of the image.
Step 2: Examine the cardiac silhouette systematically.
Step 3: Evaluate both lung fields from apex to base.
Step 4: Check the pleural spaces and costophrenic angles.
Step 5: Assess the mediastinum and hilum.
Step 6: Look at the bones and soft tissues.

After your analysis, provide:
FINDINGS: Your detailed observations
IMPRESSION: Your clinical interpretation""",
        ),
    }
    
    # Default prompts for VQA
    DEFAULT_VQA_PROMPTS = {
        "baseline": PromptTemplate(
            name="baseline",
            task="vqa",
            version="v1",
            description="Simple VQA prompt",
            system_prompt="You are a radiologist answering questions about medical images.",
            user_prompt="Question: {question}\nAnswer:",
        ),
        "detailed": PromptTemplate(
            name="detailed",
            task="vqa",
            version="v1",
            description="Detailed VQA prompt",
            system_prompt="""You are an expert radiologist. Answer questions about medical images 
accurately and concisely. For yes/no questions, respond with only 'yes' or 'no'. 
For other questions, provide a brief, accurate answer.""",
            user_prompt="""Look at this medical image carefully and answer the following question.

Question: {question}

Provide a clear, accurate answer based only on what you can observe in the image.""",
        ),
        "cot": PromptTemplate(
            name="cot",
            task="vqa",
            version="v1",
            description="Chain-of-thought VQA",
            system_prompt="You are an expert radiologist. Think carefully before answering.",
            user_prompt="""Question about this medical image: {question}

Let me analyze this step by step:
1. First, I'll identify what the question is asking about.
2. Then, I'll locate the relevant area in the image.
3. Finally, I'll provide my answer.

My answer:""",
        ),
    }
    
    # Default prompts for Grounding
    DEFAULT_GROUNDING_PROMPTS = {
        "baseline": PromptTemplate(
            name="baseline",
            task="grounding",
            version="v1",
            description="Basic grounding prompt",
            system_prompt="You are a radiologist identifying findings in medical images.",
            user_prompt="""Locate the following finding in this chest X-ray: {finding}

Provide the bounding box coordinates as [x_min, y_min, x_max, y_max] where values are normalized to 0-1.""",
        ),
        "detailed": PromptTemplate(
            name="detailed",
            task="grounding",
            version="v1",
            description="Detailed grounding with explanation",
            system_prompt="""You are an expert radiologist. Your task is to locate specific findings 
in medical images and provide precise bounding box coordinates.""",
            user_prompt="""Find and locate: {finding}

1. First, describe where you see this finding in the image.
2. Then, provide the bounding box coordinates as [x_min, y_min, x_max, y_max].
   - Coordinates should be normalized (0-1 range)
   - x_min, y_min is the top-left corner
   - x_max, y_max is the bottom-right corner""",
        ),
    }
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """Initialize with optional custom prompts directory."""
        self.prompts: Dict[str, Dict[str, PromptTemplate]] = {
            "rrg": self.DEFAULT_RRG_PROMPTS.copy(),
            "vqa": self.DEFAULT_VQA_PROMPTS.copy(),
            "grounding": self.DEFAULT_GROUNDING_PROMPTS.copy(),
        }
        
        if prompts_dir:
            self.load_custom_prompts(prompts_dir)
    
    def load_custom_prompts(self, prompts_dir: str) -> None:
        """Load custom prompts from YAML files."""
        import yaml  # Lazy import
        
        prompts_path = Path(prompts_dir)
        
        for yaml_file in prompts_path.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                
                for prompt_data in data.get("prompts", []):
                    template = PromptTemplate(
                        name=prompt_data["name"],
                        task=prompt_data["task"],
                        version=prompt_data.get("version", "v1"),
                        description=prompt_data.get("description", ""),
                        system_prompt=prompt_data["system_prompt"],
                        user_prompt=prompt_data["user_prompt"],
                        few_shot_examples=prompt_data.get("few_shot_examples", []),
                    )
                    self.add_prompt(template)
                    
                logger.info(f"Loaded prompts from {yaml_file}")
            except Exception as e:
                logger.warning(f"Failed to load {yaml_file}: {e}")
    
    def add_prompt(self, template: PromptTemplate) -> None:
        """Add a new prompt template."""
        if template.task not in self.prompts:
            self.prompts[template.task] = {}
        self.prompts[template.task][template.name] = template
    
    def get_prompt(self, task: str, name: str = "baseline") -> PromptTemplate:
        """Get a prompt template by task and name."""
        if task not in self.prompts:
            raise ValueError(f"Unknown task: {task}")
        if name not in self.prompts[task]:
            raise ValueError(f"Unknown prompt '{name}' for task '{task}'")
        return self.prompts[task][name]
    
    def list_prompts(self, task: Optional[str] = None) -> Dict[str, List[str]]:
        """List available prompts, optionally filtered by task."""
        if task:
            return {task: list(self.prompts.get(task, {}).keys())}
        return {t: list(p.keys()) for t, p in self.prompts.items()}
    
    def get_all_prompt_variations(self, task: str) -> List[PromptTemplate]:
        """Get all prompt variations for a task (for ablation studies)."""
        return list(self.prompts.get(task, {}).values())


# Example custom prompts YAML format:
EXAMPLE_PROMPTS_YAML = """
# Save this as custom_prompts.yaml in your prompts directory

prompts:
  - name: "my_custom_rrg"
    task: "rrg"
    version: "v1"
    description: "My custom RRG prompt"
    system_prompt: |
      You are a radiologist specialized in chest imaging.
    user_prompt: |
      Please analyze this chest X-ray and provide your findings.
    few_shot_examples:
      - image: "path/to/example1.jpg"
        report: "Example report 1..."
      - image: "path/to/example2.jpg"
        report: "Example report 2..."
"""
