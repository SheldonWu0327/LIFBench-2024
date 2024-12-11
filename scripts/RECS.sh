# Please enter the API_KEY in the ./evaluation/LLMApi.py file before running this script.
python ./data/RewritePrompts.py \
  --input_file './data/meta/tasks.json' \  # Manually written instructions (the first in all lists)
  --output_dir './data/meta/rewrite/' \
  --models [\'claude-3-opus-20240229\',\'gpt-4o\'] # Each model will be required to generate 20 rewrites.

python ./data/InstructionClusting.py \
  --input_dir './data/meta/rewrite/' \
  --output_dir './data/meta/cluster/' \
  --num_clusters 5 # The final number of rewrites to be generated