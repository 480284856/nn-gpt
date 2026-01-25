cd /workspace/nn-gpt

(while true; do date +"%Y-%m-%d %H:%M:%S : $(free -h | grep Mem)"; sleep 5; done > /workspace/nn-gpt/mem_usage.log) &


# nn-gpt's ab uses pkgutil.extend_path; nn-dataset must be on PYTHONPATH
# so ab.__path__ includes both ab.gpt and ab.nn (avoid ModuleNotFoundError: ab.nn.util).
export PYTHONPATH="$(cd -P /workspace/nn-dataset 2>/dev/null && pwd):${PYTHONPATH:-}"

python3 -m ab.gpt.CyclicTrainingPipeline \
    --test_nn 5 \
    --num_train_epochs 3 \
    --step4_test_prompt_num 200