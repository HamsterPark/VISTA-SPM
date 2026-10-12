"""VISTA-SPM: a visual harness for operating the simulated STM with a multimodal model.

A port of VISTA (Han, Hu, Qiu, Wu, He, "VISTA: A Visual Harness for Reasoning in an
Interactive World", arXiv 2610.02200, MIT-licensed code at github.com/joshhhhhan/VISTA)
from ARC-AGI-3 games to STM-Bench paper scenarios. The model (Claude Code in print mode,
all built-in tools disabled) sees the instrument through rendered frames, keeps a lossless
visual memory of every frame the instrument produced, can re-inspect and measure any of
them, and writes its own notes. MAST supplies the instrument skills and the safety stack;
``stmsim`` supplies the instrument; STM-Bench's judge scores the episode.

Read ``docs/DESIGN.md`` before changing anything here.
"""

__all__ = ["__version__"]
__version__ = "0.1.0.dev0"
