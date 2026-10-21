File Edit Options Buffers Tools Sh-Script Help                                                                                                                                                                                                                                                                                                    
#!/usr/bin/env bash                                                                                                                                                                                                                                                                                                                               
# 1. Start vLLM in the background                                                                                                                                                                                                                                                                                                                 
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    > $FAST/logs/vllm_server.log 2>&1 &

# 2. Wait for it to be ready (poll the health endpoint)                                                                                                                                                                                                                                                                                           
until curl -s http://localhost:8000/health > /dev/null; do
    echo "Waiting for vLLM..."
    sleep 5
done
echo "vLLM ready."

# 3. Run your analysis script in the foreground                                                                                                                                                                                                                                                                                                   
python analysis_unsupervised.py \
    -i patients/ -q analysis_questions.json -o exp/llama3.2.3b-clustering-3 \
    -m meta-llama/Llama-3.2-3B-Instruct \
    --vllm-base-url http://localhost:8000/v1 \
    --n-clusters 3

# 4. (Optional) Kill vLLM when done                                                                                                                                                                                                                                                                                                               
kill %1