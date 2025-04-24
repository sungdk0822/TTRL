from tqdm import tqdm
from ttrl.verifier.qwen.qwen_eval import qwen_reward_fn, test_time_train

def auto_verify(task, batch_size, all_outputs, all_labels):
    all_outputs = [output.outputs[0].text for output in all_outputs]

    task2verify = {
        "math": qwen_reward_fn,
        "ttt": test_time_train
    }
    assert task in task2verify, f"{task} not in {list(task2verify.keys())}"

    verify_fn = task2verify[task]
    if task in ["math"]:
        rewards = [verify_fn(output, label)
                   for output, label in tqdm(zip(all_outputs, all_labels), desc="Rewarding", total=len(all_outputs))]
    elif task in ["ttt"]:
        rewards = []
        n_prompts = len(all_outputs) // batch_size
        for prompt_idx in range(n_prompts):
            group_outputs = all_outputs[batch_size *
                                        prompt_idx:batch_size*(prompt_idx+1)]
            group_labels = all_labels[batch_size *
                                      prompt_idx:batch_size*(prompt_idx+1)]
            rewards.extend(verify_fn(group_outputs, group_labels))
    return rewards
