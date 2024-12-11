# construct the contexts $X$ in different length interval. Results will be in "{data_dir}/meta/".
python data/ConstructContext.py \
  --base_dir "." \  # Base dir of LIFBench.
  --context_lengths [3,14,28] # The length interval of the context $X$ to be constructed.

# Concatenate all components to form the prompt. Results will be in "{data_dir}/prompts/".
python data/PromptGenerate.py --base_dir '.'