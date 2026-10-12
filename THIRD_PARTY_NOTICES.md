# Third-party code in `vista_spm`

Parts of this package port the design and code of **VISTA** — Qiushi Han, Keya Hu, Linlu Qiu,
Cathy Wu, Kaiming He, "VISTA: A Visual Harness for Reasoning in an Interactive World",
arXiv 2610.02200 (2026) — from https://github.com/joshhhhhan/VISTA at commit
`c97c354f0a8948256545a3e8850a1848f8e5b8c1` (2026-10-04), released under the MIT License below.

What is derived: the stdio MCP bridge and the Claude Code segment runner, compaction hook and
checkpoint/recovery protocol (`mcp_bridge.py`, `claude_runner.py`, `compact_hook.py`), the
tool semantics and caps of `inspect` / `read_pixels` (here `inspect` / `read_values`), the
notes tools, and the wording of the system prompt (`prompts.py`). Derived files carry a one-line
attribution at the top. The STM observation layer, instrument actions, measurement tools,
episode host and ledger integration are new.

```
MIT License

Copyright (c) 2026 VISTA contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
