# OpenRouter Benchmark Harness

OpenRouter's internal benchmarking harness, externalized for transparency. We port benchmarks here so we can run them scalably on our infrastructure and iterate quickly.

```sh
bun install
OPENROUTER_API_KEY=... bun run bench -- --benchmark gpqa_diamond --model openai/gpt-4o-mini --limit 5

# Local OpenAI-compatible server (vLLM). `--model` must match the served name.
VLLM_BASE_URL=http://127.0.0.1:8000/v1 bun run bench -- --benchmark gpqa_diamond --model amd/GLM-5.3-Quark-MXFP4-AttnFP8 --limit 1 --epochs 1
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. Report security issues privately as described in [SECURITY.md](SECURITY.md).
