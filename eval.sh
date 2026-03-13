model_name="Qwen/Qwen3-VL-8B-Instruct"
base_url="http://localhost:11434/v1"
output="outputs/fvqa/qwen3_vl_8b_instruct.jsonl"

python eval_fvqa.py --model $model_name --base-url $base_url --output $output --workers 1