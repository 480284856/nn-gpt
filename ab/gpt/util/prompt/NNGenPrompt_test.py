from ab.gpt.util.prompt.NNGenPrompt import NNGenPrompt

def test_NNGenPrompt():
    prompt = NNGenPrompt(max_len=1024, tokenizer=None, prompts_path="/workspace/nn-gpt/ab/gpt/conf/prompt/train/NN_gen.json")
    prompt.get_raw_dataset(only_best_accuracy=True, n_training_prompts=10)

if __name__ == "__main__":
    test_NNGenPrompt()