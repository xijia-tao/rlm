from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "Qwen/Qwen3-VL-8B-Instruct", "base_url": "http://localhost:11434/v1", "api_key": "openai"},
    verbose=True,  # For printing to console with rich, disabled by default.
)

print(rlm.completion("Print me the first 100 powers of two, each on a newline.").response)
