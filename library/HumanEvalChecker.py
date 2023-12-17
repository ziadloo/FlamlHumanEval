import os
import json
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)
import time
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import write_jsonl
from .ResultCache import ResultCache
import math


class HumanEvalChecker:
    def __init__(self, model_directory: str,
                 training_samples: dict, pass_at_k: int,
                 seed: int, batch_size: int, average_over: int=1, cache_folder: str = "./.cache"):
        self.model_directory = model_directory
        self.seed = seed
        self.pass_at_k = pass_at_k
        self.batch_size = batch_size
        self.average_over = average_over
        self.cache_folder = cache_folder

        os.makedirs(f"{self.cache_folder}/{self.seed}/attempted_solutions", exist_ok=True)
        write_jsonl(f"{self.cache_folder}/{self.seed}/training_problems.jsonl", training_samples.values())

        self.training_samples = list(training_samples.values())

        self._initialize_model()

        self.cache = ResultCache(self.seed, cache_folder)


    def _initialize_model(self):
        self.config = ExLlamaV2Config()
        self.config.model_dir = self.model_directory
        self.config.prepare()

        self.config.scale_pos_emb = 1
        self.config.scale_alpha_value = 1

        # Tuned to max out an RTX 3090 with 24GB VRAM up to 32 samples per batch
        self.config.max_seq_len = 1024
        chunk_size = min(1024, self.config.max_seq_len)
        self.config.max_input_len = chunk_size
        self.config.max_attn_size = chunk_size ** 2

        self.model = ExLlamaV2(self.config)

        self.exlCache = ExLlamaV2Cache(self.model, batch_size = self.batch_size, lazy = True)
        self.model.load_autosplit(self.exlCache)

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        self.generator = ExLlamaV2BaseGenerator(self.model, self.exlCache, self.tokenizer)
        self.generator.warmup()


    def __call__(self, config: dict) -> dict:
        cachedResult = self.cache.check(config)
        if cachedResult is not None:
            return cachedResult

        settings = ExLlamaV2Sampler.Settings()
        if "top_p" in config["temperature_or_top_p"]:
            settings.top_p = config["temperature_or_top_p"]["top_p"]
            settings.temperature = 0.9
        else:
            settings.top_p = 0.65
            settings.temperature = config["temperature_or_top_p"]["temperature"]

        if "top_k" in config["mirostat_or_top_k"]:
            settings.top_k = config["mirostat_or_top_k"]["top_k"]
            settings.mirostat = False
        else:
            settings.mirostat = True
            settings.mirostat_eta = config["mirostat_or_top_k"]["mirostat_eta"]
            settings.mirostat_tau = config["mirostat_or_top_k"]["mirostat_tau"]
        
        settings.typical = 1

        # Apparently, HumanEval does not need more than 512 tokens to be answered
        max_new_tokens = 512

        metric_name = f"pass@{self.pass_at_k}"
        score_sum = 0
        score_sqsum = 0
        extra_info = {"tries": []}
        id = None

        time_begin = time.time()
        answers = {i: [] for i in range(self.average_over)}
        for s in range(len(self.training_samples)):
            print(f"working on sample: {s}")
            sample = self.training_samples[s]
            prompts = [sample["prompt"] for i in range(self.batch_size)]
            batches = math.ceil(self.average_over * self.pass_at_k / self.batch_size)
            generated_output = []
            for i in range(batches):
                # Generate output
                self.exlCache.current_seq_len = 0
                output = self.generator.generate_simple(prompts, settings, max_new_tokens)
                generated_output += output

            for i in range(self.average_over):
                for j in range(self.pass_at_k):
                    answers[i].append({
                        "task_id": sample["task_id"],
                        "completion": generated_output[i*self.pass_at_k + j],
                    })
        time_end = time.time()
        text_generation_time = time_end - time_begin

        for attempt in range(self.average_over):
            sample_file = f"{self.cache_folder}/{self.seed}/temp.jsonl"
            write_jsonl(sample_file, answers[attempt])

            time_begin = time.time()
            k = [1]
            if self.pass_at_k > 10:
                k.append(10)
            elif self.pass_at_k > 100:
                k.append(100)
            k.append(self.pass_at_k)
            n_workers: int = 4
            timeout: float = 3.0
            problem_file: str = f"{self.cache_folder}/{self.seed}/training_problems.jsonl"
            pass_at_k_results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
            time_end = time.time()
            answer_evaluation_time = time_end - time_begin

            ei = dict(pass_at_k_results)
            ei["text_generation_time"] = text_generation_time
            ei["answer_evaluation_time"] = answer_evaluation_time
            extra_info["tries"].append(ei)
            if id is None:
                id = self.cache.store(config, json.dumps(extra_info, sort_keys=True),
                                      metric_name, pass_at_k_results[metric_name])
            os.remove(sample_file)
            os.replace(f"{sample_file}_results.jsonl", f"{self.cache_folder}/{self.seed}/attempted_solutions/{id}-{attempt+1}.jsonl")

            score_sum = score_sum + pass_at_k_results[metric_name]
            score_sqsum = score_sqsum + pass_at_k_results[metric_name] * pass_at_k_results[metric_name]

        extra_info["score_average"] = score_sum / self.average_over
        extra_info["score_standard_deviation"] = math.sqrt(score_sqsum / self.average_over - extra_info["score_average"] * extra_info["score_average"])
        self.cache.update(id, json.dumps(extra_info, sort_keys=True), score_sum / self.average_over)

        return {metric_name: score_sum / self.average_over}
