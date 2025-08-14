# load environment variables
set -a && source .env && set +a

weco run --source optimized_inference.py \
     --eval-command "python evaluate.py" \
     --metric system_speedup \
     --goal maximize \
     --steps 5 \
     --model gpt-5 \
     --additional-instructions additional_instructions.md