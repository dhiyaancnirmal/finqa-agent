# Vendored FinQA environment code

`tools.py` and `rewards.py` are copied from Snorkel AI's FinQA
environment, the Hugging Face Space
[snorkelai/finqa-env](https://huggingface.co/spaces/snorkelai/finqa-env)
(`server/tools.py`, `server/rewards.py`, commit `055d215`, 2026-03-30; only a one-line header comment was added), which is
part of [OpenEnv](https://github.com/meta-pytorch/OpenEnv) under the BSD 3-Clause
license in `LICENSE`.

They are vendored so the tools and the grader are byte-for-byte the ones the
public environment uses, without running its Docker server.
