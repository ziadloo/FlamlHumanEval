import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from flaml import tune
from human_eval.data import read_problems
from library import HumanEvalChecker
import random


seed = 42
average_over = 2
batch_size = 32
pass_at_k = 32
training_size = 164
problems = read_problems()
problem_keys = list(problems.keys())

if os.path.exists(f".cache/{seed}/training_problems_index.json"):
    with open(f".cache/{seed}/training_problems_index.json") as f:
        training_problems_index = set(json.load(f))
else:
    training_problems_index = set(random.sample(list(range(len(problem_keys))), training_size))
    os.makedirs(f".cache/{seed}", exist_ok=True)
    with open(f".cache/{seed}/training_problems_index.json", "w") as f:
        f.write(json.dumps(list(training_problems_index)))

training_problems = {problem_keys[i]: problems[problem_keys[i]]
                     for i in range(len(problems))
                     if i in training_problems_index}
test_problems = {problem_keys[i]: problems[problem_keys[i]]
                 for i in range(len(problems))
                 if i not in training_problems_index}


model_address = "./models/TheBloke_Phind-CodeLlama-34B-v2-GPTQ"
checker = HumanEvalChecker(model_address, training_problems, pass_at_k=pass_at_k,
                           seed=seed, batch_size=batch_size, average_over=average_over)

hp_search_space = {
    "temperature_or_top_p": tune.choice(
        [
            {
                "temperature": tune.quniform(0.1, 2, 0.01)
            },
            {
                "top_p": tune.quniform(0, 1, 0.01)
            }
        ]
    ),
    "mirostat_or_top_k": tune.choice(
        [
            {
                "mirostat_eta": tune.quniform(0.005, 0.5, 0.005),
                "mirostat_tau": tune.quniform(1.0, 3.0, 0.1),
            },
            {
                "top_k": tune.qrandint(1, 500, 5)
            }
        ]
    )
}

evaluated_keys, evaluated_values = checker.cache.restore()
analysis = tune.run(
    checker, metric=f"pass@{pass_at_k}", mode="max", config=hp_search_space,
    points_to_evaluate=evaluated_keys,
    evaluated_rewards=evaluated_values,
    time_budget_s=24 * 60 * 60, num_samples=-1,
)

print(analysis.trials[-1].last_result)
