# SPO Online

### Problem description

We have n number of jobs in J, j<sub>0</sub>...j<sub>i</sub> have a due date d<sub>0</sub>...d<sub>i</sub> and predicted processing time p\*<sub>0</sub>... p\*<sub>i</sub> from features $\theta$<sub>0</sub>...$\theta$<sub>i</sub>. The actual processing time, only available after the job has finished, is defined as  p<sub>0</sub>... p<sub>i</sub>

We want to create a schedule S for a single machine (one job at a time), based on the predicted processing times in p\*, where the ordering of the allocation is defined in an array in S, and thus after running the finishing times are defined as F, f<sub>0</sub> = p<sub>0</sub> and f<sub>i</sub> = f<sub>i - 1</sub> + p<sub>i</sub>

Such that the total tardiness is minimized, total = ($\sum(i = 0...n)$ max(f<sub>i</sub> - d<sub>i</sub>, 0))

Jobs can not be preempted once started.

### Training data

CSV Format:
```
feature1, feature2, feature3, actual_processing_time
feature1, feature2, feature3, actual_processing_time
...
```

```

```
