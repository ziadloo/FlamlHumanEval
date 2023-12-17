# FlamlHumanEval
 Finding the best inference parameters for an LLM to score the HumanEval using FLAML tuning package

## Prerequisites
For this code to work, you'll need the following packages installed

* [FLAML](https://github.com/microsoft/FLAML)
  
```
pip install flaml
```

* [HumanEval](https://github.com/openai/human-eval)

```
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

* [ExllamaV2](https://github.com/turboderp/exllamav2)

```
pip install exllamav2
```

Of course, it's always advised to create a dedicated environment for your work.

## Running the code

While this repository comes with some "library" classes but it is not meant to be package but rather a sample code. You can always tweak the code in `flaml_tune.py` or write your own and make use of the classes in the "library" folder. But this repository will never be a PyPI package.

#### Since this is a machine learning exercise, having a GPU is a must.

The code written in the `flaml_tune.py` comes with a bunch of parameters tuned for an RTX 3090 with 24GBof VRAM. It is tuned so it makes the best use of the hardware while not facing an out of memory error. If you have a different hardware, you can find the best combinations of the parameters with a few trial and error.

Here are the parameters defined in the `flaml_tune.py` and what they mean:

* `average_over`: The number of times each configuration should be evaluated and averaged.
* `batch_size`: The number of samples in each batch. Increasing this will result in a better use of your hardware, as long as it won't tun our of memory.
* `pass_at_k`: The pass@k to optimize for. The steps for this parameters are: 1, 10, and 100. If you set a value like 32, lower steps (in this case 1 and 10) will also be calculated and you can find their values in the SQLite database file.
* `training_size`: How many of the HumanEval samples should be used for training. At the time of writing this, HumanEval has 164 samples.

#### Note:

Downlaod and save your LLM into a folder and mentioned the address in the `model_address` variable.

Each time you run the code, it will run for one day (24 hours). You can change this by changing the value for `time_budget_s`.

Due to a limitation of ExllamaV2, all samples within a batch should be of the same length tokewise. As the result, it's best if your `pass_at_k` to be a multiple of `batch_size`. That way, all the samples within a batch will be used and not thrown away.

You can resume an evaluation trial by keeping the `seed` value the same between two runs. The code will load the previous tries from the SQLite database and pass them on to the FLAML tune library. If you want to start a new trial, make sure you assign a new value to the seed.

While the code prints the last result of the trial, that's not necessary the best one. You can always find the metric's value for all the tries in the SQLite database and the configuration that led to it.
