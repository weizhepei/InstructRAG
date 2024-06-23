import re, json, string
from tqdm import tqdm
import numpy as np

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False

def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['answers'], item["rationale"]))

        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def get_metrics(data, save_dir=None, is_asqa=False):
    idx = 0
    num_accurate = 0
    print('Evaluating results...')
    if is_asqa:
        rationale_str_em, _ = compute_str_em(data)
    else:
        for d in tqdm(data):
            idx += 1
            is_accurate = exact_presence(d['answers'], d['rationale'])
            num_accurate += 1 if is_accurate else 0

    if is_asqa:
        print(f"Rationale EM: {rationale_str_em:.1f}%")
        eval_result = {"EM": rationale_str_em, "num_examples": idx}
    else:
        accuracy = num_accurate / idx * 100
        print(f"Accuracy: {accuracy:.1f}%")
        eval_result = {"accuracy": accuracy, "num_examples": idx}
    
    with open(f"{save_dir}/metrics.json", "w") as f:
        f.write(json.dumps(eval_result) + "\n")   

    return eval_result