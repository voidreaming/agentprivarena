from __future__ import annotations

from enum import Enum


class PrivacySchemaMode(str, Enum):
    """Controls whether privacy fields are added to write tool schemas."""

    NONE = "none"
    SELF_REPORT = "self_report"


class PrivacyAuditGuidanceMode(str, Enum):
    """Controls how read-boundary audit output is shown to the execution model."""

    INSTRUCTION_ONLY = "instruction_only"
    ANNOTATED_FLOWS = "annotated_flows"
    ANNOTATED_FLOWS_WITH_PLAN = "annotated_flows_with_plan"
    CONTEXTUAL_SHARE_DECISIONS = "contextual_share_decisions"


class PrivacyReadAuditMode(str, Enum):
    """Controls the read-boundary audit question asked by the audit model."""

    INFORMATION_FLOWS = "information_flows"
    CONTEXTUAL_SHARE = "contextual_share"


class PrivacyAuditMode(str, Enum):
    """Controls which privacy-audit mechanisms can affect execution."""

    FULL = "full"
    READ_STEER_ONLY = "read_steer_only"
    WRITE_ENFORCE_ONLY = "write_enforce_only"


class AuditPolicy(str, Enum):
    """Selects the decision criterion the audit applies to each information flow.

    The extraction and enforcement machinery is identical across policies; only
    the criterion by which a flow is judged changes. This makes the audit a
    plug-in policy engine rather than a fixed privacy rule.

    - CONTEXTUAL_INTEGRITY: (default) judge appropriateness from the relation
      among sender, recipient, subject, attribute, channel and purpose.
    - PII: judge solely on whether the flow carries third-party personally
      identifiable information, independent of recipient or purpose.
    - DATA_MINIMIZATION: judge solely on whether the flow is strictly necessary
      to accomplish the stated task, independent of sensitivity or recipient.
    """

    CONTEXTUAL_INTEGRITY = "contextual_integrity"
    PII = "pii"
    DATA_MINIMIZATION = "data_minimization"


class PrivacyStrictness(str, Enum):
    """Controls how the audit dispositions genuinely AMBIGUOUS flows.

    A privacy-utility operating point applied to the read- and write-boundary
    decisions. Clear cases are unaffected at every level (clearly appropriate
    flows always PASS; clear violations always BLOCK); this knob only sets the
    tie-break for genuinely ambiguous flows, tracing a monotone frontier:

    - LENIENT:  ambiguous -> PASS (minimal intervention, higher residual leak).
    - BALANCED: ambiguous -> ABSTRACT (generalize / re-compose). Default; the
      no-op level that reproduces prior CI-Audit behavior exactly.
    - STRICT:   ambiguous -> BLOCK (treat uncertainty as disqualifying).
    """

    LENIENT = "lenient"
    BALANCED = "balanced"
    STRICT = "strict"
