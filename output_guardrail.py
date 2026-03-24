import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gemini-2.5-flash-lite"

OUTPUT_SAFETY_CLASSIFIER_PROMPT = """Task: Classify whether the AI output contains unsafe or disallowed content.

You are a strict output safety classifier.

Classify as "yes" (UNSAFE) if the output contains:
- Actionable instructions for violence, weapons, explosives, poisoning, or physical harm
- Actionable instructions for cyber abuse, malware creation, credential theft, or illegal intrusion
- Self-harm or suicide methods
- Clear facilitation of illegal wrongdoing

Classify as "no" (SAFE) if the output is:
- High-level educational discussion without actionable harmful instructions
- Safety guidance, prevention advice, or refusal content
- Benign informational content

Output ONLY valid JSON:
{
  "label": "yes",
  "confidence": "high",
  "reasons": ["short reason"]
}

Output text to classify:
"""


def _normalize_confidence(confidence):
	if isinstance(confidence, dict):
		score = float(confidence.get("score", 0.6))
		level = str(confidence.get("level", "medium")).lower()
		return {"score": score, "level": level}

	level = str(confidence or "medium").lower()
	if level == "high":
		return {"score": 0.9, "level": "high"}
	if level == "low":
		return {"score": 0.3, "level": "low"}
	return {"score": 0.6, "level": "medium"}


def _contains_unsafe_output(text: str) -> bool:
	cleaned = re.sub(r"\s+", " ", text.lower())
	patterns = [
		r"\b(step[- ]by[- ]step|exact steps|detailed instructions)\b.*\b(bomb|explosive|grenade|poison|weapon)\b",
		r"\b(how to|best way to|instructions to)\b.*\b(hack|bypass authentication|steal credentials|deploy malware)\b",
		r"\b(how to|best way to|instructions to)\b.*\b(kill|murder|poison someone)\b",
		r"\b(suicide method|how to commit suicide|painless suicide)\b",
		r"\b(continue|next step|step \d+|as discussed above|from previous steps)\b.*\b(bomb|explosive|poison|weapon|hack|malware|kill|murder)\b",
	]
	return any(re.search(pattern, cleaned) for pattern in patterns)


def _redact_sensitive_data(text: str):
	redacted = text
	reasons = []

	redaction_rules = [
		(
			re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
			"[REDACTED_EMAIL]",
			"email address",
		),
		(
			re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
			"[REDACTED_PHONE]",
			"phone number",
		),
		(
			re.compile(r"\b(?:sk|api|key|token|secret)[-_]?[A-Za-z0-9]{12,}\b", re.IGNORECASE),
			"[REDACTED_SECRET]",
			"secret-like token",
		),
		(
			re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
			"[REDACTED_CARD]",
			"card-like number",
		),
	]

	for pattern, replacement, reason in redaction_rules:
		new_text = pattern.sub(replacement, redacted)
		if new_text != redacted:
			reasons.append(f"Redacted {reason}")
			redacted = new_text

	return redacted, reasons


class LLMOutputGuardrail:
	def __init__(self, api_key: str = None):
		self.model = None
		if api_key:
			try:
				genai.configure(api_key=api_key)
				self.model = genai.GenerativeModel(MODEL_NAME)
			except Exception:
				self.model = None

	def _classify_with_llm(self, output_text: str, prompt: str = "", conversation_history=None) -> dict:
		if not self.model:
			return {
				"label": "no",
				"confidence": {"score": 0.5, "level": "medium"},
				"reasons": ["No LLM available - deterministic checks only"],
			}

		try:
			conversation_history = conversation_history or []
			context = ""
			for i, turn in enumerate(conversation_history):
				role = turn.get("role", "unknown") if isinstance(turn, dict) else "unknown"
				content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
				context += f"Turn {i+1} ({role}): {content}\n"
			if prompt:
				context += f"Prompt: {prompt}\n"

			combined_text = output_text if not context else f"Context:\n{context}\nOutput:\n{output_text}"

			response = self.model.generate_content(
				OUTPUT_SAFETY_CLASSIFIER_PROMPT + "\n\n" + combined_text,
				generation_config=genai.types.GenerationConfig(
					temperature=0.1,
					max_output_tokens=300,
				),
			)

			raw = (response.text or "").strip()
			raw = re.sub(r"```json|```", "", raw).strip()
			parsed = json.loads(raw)

			label = str(parsed.get("label", "no")).lower().strip()
			confidence = _normalize_confidence(parsed.get("confidence", "medium"))
			reasons = parsed.get("reasons") or ["LLM output classification"]
			if not isinstance(reasons, list):
				reasons = [str(reasons)]

			return {
				"label": "yes" if label == "yes" else "no",
				"confidence": confidence,
				"reasons": reasons,
			}
		except Exception as exc:
			return {
				"label": "no",
				"confidence": {"score": 0.3, "level": "low"},
				"reasons": [f"LLM error fallback: {exc}"],
			}

	def classify_output(self, output_text: str, prompt: str = "", conversation_history=None) -> dict:
		conversation_history = conversation_history or []

		if not output_text or not output_text.strip():
			return {
				"safety": "safe",
				"blocked": False,
				"redacted": False,
				"reasons": ["Empty output"],
				"confidence": {"score": 0.7, "level": "medium"},
				"text": output_text,
			}

		if _contains_unsafe_output(output_text):
			return {
				"safety": "unsafe",
				"blocked": True,
				"redacted": False,
				"reasons": ["Deterministic unsafe output pattern"],
				"confidence": {"score": 0.95, "level": "high"},
				"text": "",
			}

		llm_safety = self._classify_with_llm(
			output_text,
			prompt=prompt,
			conversation_history=conversation_history,
		)
		if llm_safety.get("label") == "yes":
			return {
				"safety": "unsafe",
				"blocked": True,
				"redacted": False,
				"reasons": llm_safety.get("reasons", ["Unsafe output detected"]),
				"confidence": llm_safety.get("confidence", {"score": 0.8, "level": "high"}),
				"text": "",
			}

		redacted_text, redaction_reasons = _redact_sensitive_data(output_text)
		return {
			"safety": "safe",
			"blocked": False,
			"redacted": bool(redaction_reasons),
			"reasons": redaction_reasons or ["Output is safe"],
			"confidence": {"score": 0.85, "level": "high"},
			"text": redacted_text,
		}


_output_guardrail = None


def classify_output(output_text: str, prompt: str = "", conversation_history=None) -> dict:
	global _output_guardrail
	if _output_guardrail is None:
		_output_guardrail = LLMOutputGuardrail(os.getenv("GEMINI_API_KEY"))
	return _output_guardrail.classify_output(output_text, prompt, conversation_history)
