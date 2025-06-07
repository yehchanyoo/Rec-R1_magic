import lm_eval
import os
import torch

def main():
    model_name = "checkpoints/adv-ml-project/qwen2.5-3b-inst-ppo-esci_sparse-20250605_015919/actor/global_step_700"


    print(f"Starting evaluation for model: {model_name} on MBPP")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=['mbpp'], 
        batch_size=16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        confirm_run_unsafe_code=True
    )

    if results:
        print(lm_eval.utils.make_table(results))

if __name__ == "__main__":
    main()
