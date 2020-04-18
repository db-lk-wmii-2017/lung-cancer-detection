import os


def for_each_sample(path, fn):
    '''
        fn(sample, patient, patientDir)
        - sample - sample file name
        - patient - patient id 
        - patientDir - path to patient folder
    '''
    for patient in filter(lambda x: not x.startswith('.'), os.listdir(path)):
        patientDir = os.path.join(path, patient)
        for sample in os.listdir(patientDir):
            fn(sample, patient, patientDir)


def isCancerSample(sampleName):
    return int(sampleName.split('-')[1].split('.')[0]) == 1
