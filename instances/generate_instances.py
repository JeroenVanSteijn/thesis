import random

def generate_features(time):
    features = []
    for _ in range(5):
        features.append(time+random.uniform(0, 2))

    return features

def generate_processing_time(index):
    return random.randint(0, 10)

# Random instance generator
nr_instances = 5
nr_jobs_per_instance = 10

# Define general file name for instances
destination = "instances/"
processingTimes = {}
features = {}

for inst in range(nr_instances):
    for single_job_index in range(nr_jobs_per_instance):
        processingTimes[single_job_index] = generate_processing_time(single_job_index)
        features[single_job_index] = generate_features(processingTimes[single_job_index])

    with open(destination + str(inst) +'.py', 'w') as f:
        f.write("nr_jobs = ")
        f.write(str(len(processingTimes)))
        f.write('\n')
        f.write("processingTimes = ")
        f.write(str(processingTimes))
        f.write('\n')
        f.write("features = ")
        f.write(str(features))
        f.write('\n')
        f.close()
