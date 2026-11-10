"""Pipeline V4 evidence: adaptive discovery, planning and bounded execution (pass 2), hardened in pass 3.

    VERIFICATION CONTRACT (+ human-derived HF-* criteria) -> CAPABILITY DISCOVERY -> EVIDENCE PLAN
        -> BOUNDED EXECUTION (fresh evidence only; bounded replans of strategies that cannot execute)
        -> EVIDENCE BUNDLE -> CLAUDE CHECKPOINT (-> CONSISTENCY REVIEW when the answer contradicts itself)

The orchestration layer only sees generic capabilities, plans and results. Anything
technology-specific (commands, runtimes, visual capture, a browser, Unreal Engine, ...) belongs in a
CapabilityProvider (providers/), registered in registry.py.
"""
