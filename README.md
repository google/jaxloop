# Jaxloop

`jaxloop` is a custom training loop (CTL) library for Machine Learning
practitioners to train, evaluate, and checkpoint [JAX] models.

Jaxloop reduces boilerplate ML code to a standard set of modular components
(outer and inner loops, batch train/eval steps, callback actions, data/compute
partitioners) so that you can focus on modeling and experimentation. Jaxloop
components enable both common and advanced needs, from writing summaries of
metrics to distributed training ([more](#supported-tasks)) and advanced ML
techniques.

Fully control as many or as few aspects of the training loop, leveraging the JAX
ecosystem libraries you already know (`flax`, `optax`, `orbax`, etc.).

[JAX]: https://github.com/jax-ml/jax

## Installation

You can get the latest stable version of Jaxloop via

```
pip install -U jaxloop
```

<!--
To use the development version (or to test [contributions](CONTRIBUTING)), clone
this repo and run

```
...
```
-->

## Supported tasks

Jaxloop includes a growing list of out-of-box features.

* Highly customizable **training loop**
* Automatic model **evaluation**
* Epoch-level **actions** (every given number of train/eval steps)
  + Metric aggregation and summary writing (reports, logging)
  + Loading or exporting model checkpoints
  + Advanced ML techniques e.g. learning rate scheduling, early stopping, etc.
* Built-in **partitioners** for distributed data processing and training on
  different device topologies (CPU, GPU, and TPU)

## Basic usage

```
import jaxloop
import optax

...

# Define training and evaluation steps

# Initializations

config = ...  # hyperparameters
prng = ...  # JAX random numbers
model, batch = ...  # model architecture
training_split, steps_per_epoch = ...
evaluation_split = ...  # dataset splits

learning_rate = config.get("training.learning_rate")
train_step = step.Step(
    base_prng=prng,
    model=model,
    optimizer=optax.adam(learning_rate),
    train=True,
)
train_state = train_step.initialize_model(batch)

eval_step = step.Step(
    base_prng=prng,
    model=model,
    train=False,
)

# Define the inner training and evaluation loops

train_loop = train_loop_lib.TrainLoop(
  train_step, end_actions=training_end_actions
)

eval_loop = eval_loop_lib.EvalLoop(eval_step)

# Define the outer loop and run it

outer_loop = outer_loop_lib.OuterLoop(
    train_loop=train_loop, eval_loops=[eval_loop]
)
eval_spec = outer_loop.EvalSpec(evaluation_split)

num_epochs = config.get("training.num_epochs", 100)
train_steps = num_epochs * steps_per_epoch

outer_loop(
    train_state,
    train_dataset=iter(training_split),
    train_total_steps=train_steps,
    train_loop_steps=config["training.steps_per_loop"],
    eval_specs=[eval_spec],
    **config["training.kwargs"],
)
```