"""
ClinicalStructuringAgent: turn redacted note + NER entities into structured clinical state.

Uses NER outputs only (no extra LLM). Adds deterministic regex vitals parsing and
clinical heuristics for missing critical information.
"""

from __future__ import annotations

import re
from typing import Any

from services.shared.schemas_v1 import EntityItem


class ClinicalStructuringAgent:
    """Builds ``structured_features``, ``signals``, and ``missing_inputs`` from text + entities."""

    _HEDGE = re.compile(
        r"\b(?:maybe|possibly|perhaps|unclear|unknown|unsure|not sure|"
        r"questionable|apparently|seems?|might|could be|vs\.?|differential)\b",
        re.I,
    )
    _DURATION = re.compile(
        r"\b(?:for|since|about|approximately|x)\s+\d{1,4}\s*(?:"
        r"min|mins|minutes?|hrs?|hours?|h\b|days?|wks?|weeks?|months?|mos?|yrs?|years?)\b"
        r"|\b\d{1,4}\s*(?:min|mins|minutes?|hrs?|hours?|h\b|days?|weeks?|months?|years?)\b"
        r"|\bsince\s+(?:yesterday|last week|childhood|birth|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
        r"|\b(?:chronic|acute|intermittent|constant|recurrent|sudden|gradual|life.?long)\b",
        re.I,
    )
    _SEVERITY = re.compile(
        r"\b(?:mild|moderate|severe|excruciating|debilitating|worst|"
        r"\d\s*/\s*10|/10|out of 10|nrs|vas)\b"
        r"|\bpain\s+(?:is\s+)?(?:\d+/\d+|\d+ out of)\b",
        re.I,
    )
    _AGE = re.compile(
        r"(?:\b|^)(?:age|aged)\s*[:\s]?\s*(\d{1,3})\b"
        r"|\b(\d{1,3})\s*(?:y\.?o\.?|year|yr)s?\s*(?:old|of age)\b"
        r"|\b(\d{1,3})\s*yo\b",
        re.I,
    )
    _BP = re.compile(
        r"(?:bp|blood pressure)\s*[:\s]\s*(\d{2,3})\s*/\s*(\d{2,3})"
        r"|(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b",
        re.I,
    )
    _HR = re.compile(
        r"(?:hr|heart rate|pulse)\s*[:\s]\s*(\d{2,3})\b|\bpulse\s+(\d{2,3})\b",
        re.I,
    )
    _SPO2 = re.compile(
        r"(?:spo2|o2\s*sat(?:uration)?|oxygen\s*sat)\s*[:\s]\s*(\d{2,3})\s*%?\b",
        re.I,
    )
    _TEMP = re.compile(
        r"(?:temp(?:erature)?|t\.?)\s*[:\s]\s*(\d{2}(?:\.\d+)?)\s*°?\s*([fc]?)\b"
        r"|(\d{2}(?:\.\d+)?)\s*°\s*([fc])\b",
        re.I,
    )
    _RISK_SMOKING = re.compile(
        r"\b(?:current|active)\s+smoker|smokes?\s+(?:daily|regularly)|tobaccos?\s+use:?\s*yes|pack[\s-]*(?:year|yr)s?\b",
        re.I,
    )
    _AC = re.compile(
        r"\b(?:on\s+)?(?:warfarin|coumadin|apixaban|eliquis|rivaroxaban|xarelto|"
        r"dabigatran|pradaxa|edoxaban|savaysa|clopidogrel|plavix)\b",
        re.I,
    )

    @classmethod
    def enrich(
        cls,
        redacted_text: str,
        entities: list[EntityItem],
        *,
        pii_redacted: bool,
    ) -> dict[str, Any]:
        """
        Return payload extensions: structured_features, signals, missing_inputs, clinical groupings.

        ``structured_features`` aligns with scoring-service numeric/flag rules where possible.
        """
        text = redacted_text or ""
        normalized_types = [e.type.upper().strip() for e in entities]

        symptoms, diseases, risk_factors = cls._bucket_entities(entities)
        vitals = cls._extract_vitals(text)
        structured_features: dict[str, Any] = {k: v for k, v in vitals.items() if v is not None}

        age_val = cls._extract_age(text, entities)
        if age_val is not None:
            structured_features["age"] = float(age_val)

        if cls._RISK_SMOKING.search(text):
            structured_features["smoking_current"] = True
        if cls._AC.search(text):
            structured_features["on_anticoagulant"] = True

        hedging = bool(cls._HEDGE.search(text))
        duration_ok = bool(cls._DURATION.search(text))
        severity_ok = bool(cls._SEVERITY.search(text))

        has_clinical_focus = bool(symptoms or diseases or len(text) > 40)
        needs_severity_context = bool(symptoms) or cls._painlike(text)

        missing_inputs: list[str] = []
        if pii_redacted and has_clinical_focus and age_val is None and "AGE" not in normalized_types:
            missing_inputs.append("age")
        if pii_redacted and (symptoms or diseases) and not duration_ok:
            missing_inputs.append("duration")
        if pii_redacted and needs_severity_context and not severity_ok:
            missing_inputs.append("severity")

        signals: dict[str, Any] = {
            "uncertainty": {
                "hedging_language": hedging,
                "vague_timing": (symptoms or diseases) and not duration_ok,
                "severity_unspecified": needs_severity_context and not severity_ok,
            },
            "vitals": {
                "blood_pressure_parsed": vitals.get("systolic_bp") is not None
                and vitals.get("diastolic_bp") is not None,
                "any_vital_extracted": any(
                    v is not None for k, v in vitals.items() if k in {"systolic_bp", "heart_rate", "spo2"}
                ),
            },
            "clinical_buckets": {
                "symptoms": [e.model_dump() for e in symptoms],
                "diseases": [e.model_dump() for e in diseases],
                "risk_factors": [e.model_dump() for e in risk_factors],
            },
        }

        confidence = cls._score_confidence(
            pii_redacted=pii_redacted,
            entity_count=len(entities),
            vitals=structured_features,
            missing_inputs=len(missing_inputs),
            hedging=hedging,
            buckets=(len(symptoms), len(diseases), len(risk_factors)),
        )

        return {
            "structured_features": structured_features,
            "signals": signals,
            "missing_inputs": missing_inputs,
            "structuring_confidence_hint": confidence,
        }

    @staticmethod
    def _painlike(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:pain|ache|hurts|tender|cramp|pressure|discomfort)\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _bucket_entities(
        entities: list[EntityItem],
    ) -> tuple[list[EntityItem], list[EntityItem], list[EntityItem]]:
        symptoms: list[EntityItem] = []
        diseases: list[EntityItem] = []
        risk_factors: list[EntityItem] = []
        for e in entities:
            t = e.type.upper().strip()
            if "SYMPTOM" in t or t in {"SIGN", "COMPLAINT"}:
                symptoms.append(e)
            elif "DISEASE" in t or t in {"CONDITION", "DIAGNOSIS", "DX", "PROBLEM"}:
                diseases.append(e)
            elif (
                "RISK" in t
                or "COMORBID" in t
                or t in {"RISK_FACTOR", "SOCIAL", "FAMILY_HISTORY", "MEDICATION", "DRUG"}
            ):
                risk_factors.append(e)
            elif any(k in t for k in ("LAB", "VITAL", "PROCEDURE", "ALLERGY")):
                risk_factors.append(e)
            elif any(k in t for k in ("PAIN", "SYM", "NAUSEA", "FEVER", "SOB", "DYSP")):
                symptoms.append(e)
            elif not t:
                diseases.append(e)
            else:
                diseases.append(e)
        return symptoms, diseases, risk_factors

    @classmethod
    def _extract_age(cls, text: str, entities: list[EntityItem]) -> int | None:
        m = cls._AGE.search(text)
        if m:
            for g in m.groups():
                if g is not None:
                    try:
                        age = int(g)
                        if 0 <= age <= 120:
                            return age
                    except ValueError:
                        continue
        for e in entities:
            if e.type.upper().strip() in {"AGE", "PATIENT_AGE", "DEMOGRAPHIC"}:
                found = re.findall(r"\d+", e.text)
                if found:
                    try:
                        age = int(found[0])
                        if 0 <= age <= 120:
                            return age
                    except ValueError:
                        continue
        return None

    @classmethod
    def _extract_vitals(cls, text: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "systolic_bp": None,
            "diastolic_bp": None,
            "heart_rate": None,
            "spo2": None,
            "temperature_c": None,
        }
        m_bp = cls._BP.search(text)
        if m_bp:
            groups = [g for g in m_bp.groups() if g is not None]
            if len(groups) >= 2:
                try:
                    s, d = int(groups[0]), int(groups[1])
                    if 50 <= s <= 300 and 30 <= d <= 200:
                        out["systolic_bp"] = float(s)
                        out["diastolic_bp"] = float(d)
                except (TypeError, ValueError):
                    pass

        m_hr = cls._HR.search(text)
        if m_hr:
            g = next((x for x in m_hr.groups() if x is not None), None)
            if g is not None:
                try:
                    hr = int(g)
                    if 20 <= hr <= 250:
                        out["heart_rate"] = float(hr)
                except ValueError:
                    pass

        m_o2 = cls._SPO2.search(text)
        if m_o2:
            try:
                spo2 = int(m_o2.group(1))
                if 50 <= spo2 <= 100:
                    out["spo2"] = float(spo2)
            except (ValueError, IndexError):
                pass

        m_t = cls._TEMP.search(text)
        if m_t:
            try:
                val_g = m_t.group(1) or m_t.group(3)
                unit = (m_t.group(2) or m_t.group(4) or "f").upper()
                if val_g:
                    val = float(val_g)
                    if unit == "F":
                        temp_c = (val - 32.0) * 5.0 / 9.0
                    else:
                        temp_c = val
                    if 30.0 <= temp_c <= 44.0:
                        out["temperature_c"] = round(temp_c, 2)
            except (ValueError, IndexError, TypeError):
                pass

        return out

    @staticmethod
    def _score_confidence(
        *,
        pii_redacted: bool,
        entity_count: int,
        vitals: dict[str, Any],
        missing_inputs: int,
        hedging: bool,
        buckets: tuple[int, int, int],
    ) -> float:
        if not pii_redacted:
            return 0.22
        score = 0.48
        if entity_count:
            score += min(0.22, 0.06 + 0.03 * min(entity_count, 5))
        sbuck = sum(1 for x in buckets if x > 0)
        score += min(0.12, 0.04 * sbuck)
        vital_hits = sum(
            1 for k in ("systolic_bp", "diastolic_bp", "heart_rate", "spo2") if vitals.get(k) is not None
        )
        if vital_hits:
            score += min(0.1, 0.03 * vital_hits)
        score -= min(0.24, 0.08 * missing_inputs)
        if hedging:
            score -= 0.06
        return max(0.05, min(0.98, score))
