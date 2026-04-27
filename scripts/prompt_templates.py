from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PromptStage:
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int


class MedicalPromptTemplates:
    def __init__(self):
        self.stages: Dict[str, PromptStage] = {
            "evidence_evaluator": PromptStage(
                name="证据评估器",
                system_prompt=(
                    "You are a medical evidence summarizer. "
                    "Use only the provided context. "
                    "Return one short plain-text sentence only."
                ),
                user_prompt_template=(
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n\n"
                    "Task:\n"
                    "Write one short sentence summarizing the most relevant evidence. "
                    "Do not explain your reasoning. Do not use JSON."
                ),
                temperature=0.0,
                max_tokens=60,
            ),
            "answer_generator": PromptStage(
                name="答案生成器",
                system_prompt=(
                    "You are a careful medical QA assistant. "
                    "Use only the provided context and evidence summary. "
                    "Be conservative and evidence-grounded. "
                    "Do not generalize beyond the specific populations, outcomes, or study settings mentioned. "
                    "If evidence is indirect, limited, mixed, or population-specific, say so explicitly. "
                    "Do not overclaim benefit, safety, or causality. "
                    "Return plain text only. Do not use JSON. Do not explain your reasoning."
                ),
                user_prompt_template=(
                    "Question:\n{question}\n\n"
                    "Evidence summary:\n{evidence_summary}\n\n"
                    "Filtered context:\n{context}\n\n"
                    "Task:\n"
                    "Answer the question in 3 to 4 short sentences.\n"
                    "Requirements:\n"
                    "1. Use only the provided evidence.\n"
                    "2. Do not extend findings from a specific subgroup to all patients.\n"
                    "3. Distinguish direct cardiovascular evidence from indirect cerebrovascular or cognitive findings.\n"
                    "4. If the evidence is limited or population-specific, say so clearly.\n"
                    "5. Avoid phrases like 'proves', 'clearly shows', or broad claims of protection.\n"
                    "6. Plain text only. No JSON.\n"
                ),
                temperature=0.0,
                max_tokens=140,
            ),
            "critical_reviewer": PromptStage(
                name="批判性审查器",
                system_prompt=(
                    "You are a medical reviewer. "
                    "Return one short plain-text sentence only."
                ),
                user_prompt_template=(
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n\n"
                    "Draft answer:\n{draft_answer}\n\n"
                    "Task:\n"
                    "State in one short sentence whether the answer is acceptable."
                ),
                temperature=0.0,
                max_tokens=40,
            ),
            "final_assembler": PromptStage(
                name="最终组装器",
                system_prompt=(
                    "You are a medical answer editor. "
                    "Return short plain text only."
                ),
                user_prompt_template=(
                    "Question:\n{question}\n\n"
                    "Draft answer:\n{draft_answer}\n\n"
                    "Review feedback:\n{review_feedback}\n\n"
                    "Task:\n"
                    "Write the final answer in 3 to 4 short sentences. "
                    "Do not use JSON."
                ),
                temperature=0.0,
                max_tokens=140,
            ),
        }

    def get_stage(self, stage_name: str) -> PromptStage:
        if stage_name not in self.stages:
            raise KeyError(f"Unknown prompt stage: {stage_name}")
        return self.stages[stage_name]

    def render(self, stage_name: str, **kwargs: Any) -> Dict[str, Any]:
        stage = self.get_stage(stage_name)
        return {
            "name": stage.name,
            "system_prompt": stage.system_prompt,
            "user_prompt": stage.user_prompt_template.format(**kwargs),
            "temperature": stage.temperature,
            "max_tokens": stage.max_tokens,
        }