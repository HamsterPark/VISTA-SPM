"""STM-BabyVision: a static perception benchmark built on the STM-Bench simulator (VISTA-SPM WP-D).

Frames with known truth are generated on the simulator (:mod:`.generate`), each paired with a
question in the BabyVision style (:mod:`.items` holds the JSONL schema). A question is answered
through the same visual-memory and measurement tools a VISTA episode has, with no instrument
action (:mod:`.qa_host`), or by the classical reference answerer (:mod:`.baseline`); answers are
scored per type with explicit tolerances (:mod:`.score`) and tabulated (:mod:`.report`).

CLI: ``python -m vista_spm.staticbench {generate,validate,answer,score,report,qa} ...`` —
see ``docs/staticbench.md``.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
