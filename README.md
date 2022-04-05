# Thesis

## Installation

### Installing spo-online

```
cd spo-online
pip3 install -r requirements.txt
```

### Running spo-online

```
cd spo-online
python3 spo-online.py
```

## TODO List

- Implement a simpler MIP
    - Generate small instances so that training is fast.
- Implement SPO loss
    - Find some way to generate feature data that the ML can train on which is not as simple as a uniform distribution around the actual processing time.
- Implement SPO plus loss
- Implement SPO online addition (simulate online optimization in the training)

### Implementation questions

- How do we train for a single prediction when it's actually a combination of the predictions that lead to the decision error.
- Can we calculate the decision error of the prediction version through the MIP somehow? I don't want to calculate the cost manually for every MIP problem.
