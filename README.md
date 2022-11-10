# Predict+optimize for Combinatorial Optimization with uncertainty in the constraints

## Installation

```bash
cd uncertain-weights-self-generated
pip3 install -r requirements.txt
```

## Running replication experiment

The replication of the 0-1 knapsack problem with uncertain weights is based on the following Github repository: https://github.com/JayMan91/aaai_predit_then_optimize

The experiment can be replicated with the following commands:

```bash
cd replication
make run
```

## Running experiments that mirror this experiment to a setting with uncertain weights

This repository was altered slightly to accomodate for a similar problem with uncertainty in the weights instead of the values.

```bash
cd uncertain-weights-mirrored-from-replication
make run
```

## Running the experiments of this thesis

In `uncertain-weigths-self-generated`, the experiments are located that are used for gathering the final results of the thesis.

### Instance generation

Instances can be generated with the following commands:

```bash
cd uncertain-weights-self-generated
make generate
```

After this the instance is located in `./instances/{foldername}`. The `foldername` is a variable in the procedure.

The instance generation procedure can be altered and tweaked in `uncertain-weights-self-generated/generate_instances.py`

Options are located at the top of the file and include:

```python
nr_seeds = 5 # The number of times to repeat the procedure on different seeds.
nr_items = 100804 # The number of knapsack items to generate
weight_value_ratio = 0.2 # The weight to value ratio of each item.
noiseSize = 20 # The amount of noise to add to the feature data.

foldername = f"linear_combination_20_noise" # The folder name to store the instances in.
min_value = 10 # The minimum value.
max_value = 50 # The maximum value.

generate_multiple_realizations_small_sample = False # Multiple realizations experiment data.
generate_from_knap_pi_file_example = False # Instances based on the 'knappi' instances.
generate_from_energy_file_example = True # Instances based on the ICON energy price data.
# Otherwise: use a linear combination to generate the weights from a uniform distribution.
```

### Running the experiments on the instance data.

The experiments can be run with the following commands:

```bash
cd uncertain-weights-self-generated
make run
```

The experiment variables are located in `uncertain-weights-self-generated/main.py`

Options are located at the top of the file and include:

The selected learning methods are located at the bottom of the file:

```python
    learner = SGD_SPO_dp_lr(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        n_items=n_items,
        capacity=[capacity],
        penalty_P=1000, # Only required for linear_values and linear_weights penalty methods.
        penalty_function_type="linear_weights", # Can be 'repair', 'reject', 'linear_values' and 'linear_weights'
        file_name=folder + "/spo_learner_p1000_linear_weights.py", # Where to store the results.
    )
    learner.fit(x_train, y_train, x_validation, y_validation)

    learner = MSE_Learner(
        values_train=values_train,
        values_validation=values_validation,
        epochs=epochs,
        optimizer=optim.Adam,
        capacity=[capacity],
        n_items=n_items,
        file_name=folder + "/mse_learner.py", # Where to store the results.
    )
    learner.fit(x_train, y_train, x_validation, y_validation)
```

### Plotting the results

3 types of plotting are included:

- `make plot` (located in `plot.py`) was used to plot any other data and altered on-the-fly. It is most similar to plot_to_screen but saves to an image file instead. It is not strictly necessary.
- `make plots` (located in `plot_to_screen.py`)  plots the results for multiple learners on a single type of instance (f.e. a linear combination with 20 noise) to the screen.
- `make plotf` (located in `plot_outcome_overview.py`) plots the results of multiple learners between multiple types of instance files (f.e. varied noise or varied amounts of training data).

### File structure

- `energy/` -> provides original ICON energy data.
- `images/` -> stores plot images
- `instances/` -> provides the generated instances to run experiments on.
- `knapPI/` -> provides original knapsack data for the experiment in Psinger.
- `results/` -> stores the experiment results
- `csv_writer.py` -> writes the results to a CSV file.
- `generate_instances.py` -> generates new instances
- `learner.py` -> general learning methods relevant to both MSE and SPO.
- `main.py` -> starts running the experiments and configures them.
- `Makefile` -> contains the bash commands
- `MSE_learner.py` -> contains th methods for MSE based learning
- `plot_outcome_overview.py` -> related to plotting
- `plot_to_screen.py` -> related to plotting
- `plot.py` -> related to plotting
- `solving.py` -> contains the GurobiPy MIP solver for the 0-1 knapsack problem
- `SPO_learner.py` -> contains the methods for SPO-based learning
- `requirements.txt` -> contains the installation packages required for Python (pip)

### Branches

- `main` is the final branch used for gathering the results in the thesis.
- `feature/data-set-eliza` contains all alterations to the main branch required to gather the results for Appendix E in which Branch & Learn - C was compared to the methods of this work.