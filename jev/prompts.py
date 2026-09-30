"""Versioned Jev questions. Selection uses training-split development cases."""

import copy
import json
from pathlib import Path

BASIC = {
    "decision": {
        "type": "choice",
        "instructions": (
            "You are reviewing an independent medical review (IMR) appeal of a "
            "health insurance coverage denial. Based only on the case description, "
            "predict whether the insurer's denial was upheld or overturned on appeal."
        ),
        "criteria": {
            "Upheld": "The independent review upheld the insurer's coverage denial.",
            "Overturned": "The independent review overturned the insurer's coverage denial.",
        },
    }
}

STRUCTURED = {
    "decision": {
        "type": "choice",
        "instructions": {
            "task": (
                "Predict the actual final outcome of an independent medical review (IMR) "
                "appeal of a health insurance coverage denial from the supplied case description."
            ),
            "review_factors": [
                "Identify the specific requested service, treatment, medication, test, or level of care.",
                "Assess the diagnosis, symptom severity, functional impairment, and clinical risk described.",
                "Consider prior treatments, their response, contraindications, and suitable alternatives.",
                "Identify the insurer's stated denial rationale and assess the case facts supporting or contradicting it.",
                "For medical necessity, assess whether the requested service, setting, intensity, and duration fit the clinical need.",
                "For investigational or experimental denials, assess evidence of effectiveness and applicability to this indication.",
                "Apply coverage terms or eligibility conditions only when the description supplies them.",
            ],
            "interpretation_rules": [
                "The original insurer denial is the starting point of every appeal; it is not the final review decision.",
                "Predict what the independent reviewer likely decided, rather than what the patient or insurer wants.",
                "A serious diagnosis alone does not establish that this particular requested service is appropriate.",
                "A treatment being nonstandard alone does not establish that it lacks evidence for this patient.",
                "Do not invent patient history, failed treatments, policy exclusions, or clinical findings.",
                "A detail omitted from a short description is unknown; omission alone is not evidence against coverage.",
                "Consider the evidence in context. The signals below are considerations, not automatic decision rules.",
                "When the facts are insufficient, still choose the more likely outcome and reflect ambiguity in the probabilities.",
            ],
        },
        "criteria": {
            "Upheld": {
                "meaning": "The independent review sustains the insurer's denial of the requested coverage.",
                "supporting_signals": [
                    "The requested service or level of care is not justified by the clinical severity or needs described.",
                    "The requested care lacks demonstrated benefit or evidence for this indication.",
                    "The facts support the stated investigational, medical-necessity, or explicitly supplied coverage rationale.",
                    "A suitable alternative or less intensive setting meets the patient's described needs.",
                ],
                "boundary": "The mere presence of an initial denial or absence of detail is insufficient to favor this outcome.",
            },
            "Overturned": {
                "meaning": "The independent review reverses the insurer's denial in favor of the requested coverage.",
                "supporting_signals": [
                    "The requested care is appropriate and medically necessary for the clinical condition described.",
                    "Severity, risk, or impairment supports the requested intensity or setting of care.",
                    "Relevant alternatives have failed, are contraindicated, or cannot meet the described clinical need.",
                    "Evidence supports benefit for this indication or case facts undermine the insurer's stated denial rationale.",
                ],
                "boundary": "Patient preference, a diagnosis, or prior treatment alone is insufficient to favor this outcome.",
            },
        },
    }
}

FOCUSED = {
    "decision": {
        "type": "choice",
        "instructions": {
            "question": "Was the health insurer's coverage denial UPHELD or OVERTURNED in independent medical review?",
            "focus": (
                "Predict the review outcome for the current case. Assess whether the particular "
                "requested treatment, test, drug, or care setting is medically necessary and "
                "supported for the described condition. Consider clinical severity, functional "
                "needs, prior treatment response, reasonable alternatives, and the denial rationale."
            ),
            "cautions": (
                "The insurer's initial denial is not the review outcome. Do not assume missing "
                "facts are negative findings or invent patient history or coverage restrictions. "
                "Serious illness alone does not justify every intervention. Failed alternatives "
                "alone do not establish benefit of an investigational treatment."
            ),
        },
        "criteria": {
            "Upheld": (
                "Review agrees with the denial: the specific requested care is not medically "
                "necessary or appropriately supported for this case, its setting or intensity "
                "is not warranted, or evidence supports the stated investigational or coverage rationale."
            ),
            "Overturned": (
                "Review disagrees with the denial: the requested care is justified by the patient's "
                "clinical needs, prior treatment history, and evidence of benefit for this indication, "
                "or those facts undermine the stated denial rationale."
            ),
        },
    }
}

FOCUSED_FEWSHOT = copy.deepcopy(FOCUSED)
FOCUSED_FEWSHOT["decision"]["instructions"]["labeled_examples"] = [
    {"case": example["text"], "review_outcome": example["decision"].title()}
    for example in json.loads(Path(__file__).with_name("fewshot_examples.json").read_text())
]
FOCUSED_FEWSHOT["decision"]["instructions"]["example_scope"] = (
    "The labeled examples illustrate past reviews. Classify only the current case "
    "in state; example facts are not facts about the current patient."
)

PROMPTS = {
    "basic": BASIC, "structured": STRUCTURED,
    "focused": FOCUSED, "focused-fewshot": FOCUSED_FEWSHOT,
}
DEFAULT_PROMPT = "focused"  # Selected on 100 train-split dev cases; ties favor zero-shot.
