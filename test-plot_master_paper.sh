cd /workspace/nn-gpt

# nn-gpt's ab uses pkgutil.extend_path; nn-dataset must be on PYTHONPATH
# so ab.__path__ includes both ab.gpt and ab.nn (avoid ModuleNotFoundError: ab.nn.util).
export PYTHONPATH="$(cd -P /workspace/nn-dataset 2>/dev/null && pwd):${PYTHONPATH:-}"

python3 -m ab.gpt.CyclicTrainingPipeline \
    --num_cycles 1 \
    --test_nn 4 \
    --num_train_epochs 1 \
    --step4_test_prompt_num 1