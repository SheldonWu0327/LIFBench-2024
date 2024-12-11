# You can find all the outputs in ./data/outputs

model_names=(
  "./models/Qwen2.5-7B-Instruct/" \
  "./models/Qwen2.5-7B/" \
  "./models/Qwen2.5-14B/" \
  "./models/Qwen2.5-14B-Instruct/" \
  "./models/c4ai-command-r-v01" \
  "./models/Qwen2.5-32B-Instruct" \
  "./models/Qwen2.5-72B-Instruct/" \
  "./models/Qwen2.5-32B" \
  "./models/Qwen2.5-72B/" \
  "./models/glm-4-9b-chat-1m" \
  "./models/Meta-Llama-3.1-8B" \
  "./models/Meta-Llama-3.1-8B-Instruct" \
  "./models/internlm2_5-7b-chat-1m/" \
  "./models/LWM-Text-1M" \
  "./models/LWM-Text-Chat-1M" \
  "./models/c4ai-command-r-v01" \
  "./models/c4ai-command-r-08-2024" \
)


for model_name in "${model_names[@]}"; do
  python ./evaluation/inference.py --model_name "$model_name" --benchmark_base_dir '.'
  # capture the return
  return_value=$?
  echo "========================== Python script first returned: $return_value =========================="
  # Use a while loop to repeatedly run the script until the return value is 0.

  while [ $return_value -ne 0 -a $return_value -le 10 ]; do
    # Run the Python script and pass the current parameters.
    python ./evaluation/inference.py --model_name "$model_name" --seqlen_reduce_ratio "$return_value" --benchmark_base_dir '.'
    return_value=$?
    echo "==========================Python script returned: $return_value =========================="
  done
done
