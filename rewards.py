
import re
import weave
from math_verify import parse, verify, ExprExtractionConfig

from utills import get_wandb_run, wandb_attributes


@weave.op # <- this the logging we care
def reward_one_correct(one_response, one_answer):
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, one_response)
    if len(nums) == 0:
        return -1.0
    lastnum = nums[-1]
    try:
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(one_answer, extraction_config=[ExprExtractionConfig()])
        return 1.0 if verify(ans, ground_truth) else -1.0
    except:
        return -1.0

@weave.op
def reward_correct(completions, answer, **kwargs):
    """Verify if the completions is mathematically correct"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response, answer in zip(responses, answer):
        with wandb_attributes():
            reward = reward_one_correct(response, answer)
        rewards.append(reward)
    return rewards


@weave.op # <- this the logging we care
def reward_one_format(one_response):
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    think_count = one_response.count("<think>") + one_response.count("</think>")
    answer_count = one_response.count("<answer>") + one_response.count("</answer>")
    return (
        1.25
        if re.match(pattern, one_response, re.DOTALL | re.VERBOSE)
        and think_count == 2
        and answer_count == 2
        else -1.0
    )


@weave.op
def reward_format(completions, **kwargs):
    """Verify if the completions follow the required format"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        with wandb_attributes():
            reward = reward_one_format(response)
        rewards.append(reward)
    return rewards
