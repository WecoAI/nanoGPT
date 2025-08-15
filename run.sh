# load environment variables
set -a && source .env && set +a

# run the optimization
weco run --source optimized_inference.py \
     --eval-command "python evaluate.py" \
     --metric system_generation_time \
     --goal minimize \
     --steps 100 \
     --model gpt-5 \
     --additional-instructions cuda_guide.md
