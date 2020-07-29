# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : utils.py
#
#* Purpose :
#
#* Creation Date : 02-11-2018
#
#* Last Modified : Fri 23 Nov 2018 01:02:53 PM EST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#
# Cannibalized from https://github.com/anandjeyahar/data-science-utils/blob/master/datascienceutils/utils.py
# Might be better as part of toolbelt
import csv
import datetime
import json
import os

from sklearn.externals import joblib

from toolbelt import settings


def get_full_path(base_path, filename, model_params, extn='.pkl', params_file=False):
    if params_file:
        return os.path.join(base_path, '{}_{}_params.json'.format(model_params['id'], filename))
    else:
        return os.path.join(base_path, '{}_{}'.format(model_params['id'], filename + extn))

def dump_model(model, filename, model_params):
    """
    @params:
        @model: actual scikits-learn (or supported by sklearn.joblib) model file
        @model_params: parameters used to build the model
        @filename: Filename to store the model as.
    @return:
        Returns the file name
    @output:
        Dumps the model and the parameters as separate files
    """
    import uuid
    from sklearn.externals import joblib

    assert model, "Model required"
    assert filename, "Filename Required"
    assert model_params, "model parameters (dict required)"
    assert 'model_type' in model_params, "model_type required in model_params"
    assert 'output_type' in model_params, "output_type required in model_params"
    assert 'input_metadata' in model_params, "input_metadata required in model_params"
    if not os.path.exists(settings.MODELS_BASE_PATH):
        os.mkdir(settings.MODELS_BASE_PATH)
    if not 'output_metadata' in model_params:
        model_params['output_metadata'] = None
    if not 'batch_size' in model_params:
        model_params['batch_size'] = 'na'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    model_params.update({'id': timestamp})

    if any(map(lambda x: x in model_params['model_type'], ['keras', 'tensorflow'])):
        model_file = get_full_path(settings.MODELS_BASE_PATH, filename, model_params, extn='.h5')
        model.save(model_file)
    else:
        model_file = get_full_path(settings.MODELS_BASE_PATH, filename, model_params, extn='.pkl')
        joblib.dump(model, model_file, compress=('lzma', 3))


    # Save the parameters
    with open(get_full_path(settings.MODELS_BASE_PATH, filename,
                            model_params, extn='.json', params_file=True), 'w') as params_file:
        json.dump(model_params, params_file)
    # Update models performance comparison csv
    model_params.update({'modelfile': model_file})
    perf_tracker = ModelPerfTracker(model_file, model_params)
    perf_tracker.initialize()
    perf_tracker.update_model_performance()
    return model_file

class ModelPerfTracker:
    def __init__(self, modelname, params, **kwargs):
        self.modelname = modelname
        self.params=params
        # Ensure minimum parameters are present
        for p in ['modelname', 'modelfile', 'score', 'batch_size', 'epochs']:
            if p not in params:
                raise ValueError("params is missing required element {p}".format(p=p))
        for k,v in params.items():
            if k not in ['model_confs', 'output_type', '']:
                setattr(self,k, v)

    def initialize(self):
        if not os.path.exists(settings.MODEL_PERF_TRACKER):
            with open(settings.MODEL_PERF_TRACKER, 'w') as csvfile:
                fieldnames = settings.MODEL_PERF_FIELDS
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def update_model_performance(self):
        with open(settings.MODEL_PERF_TRACKER, 'a') as csvfile:
            fieldnames = settings.MODEL_PERF_FIELDS
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'timestamp': datetime.datetime.utcnow().isoformat(),
                             'modelname': self.modelname,
                             'modelfile': self.modelfile,
                             'validation_loss': self.score[0],
                             'validation_accuracy': self.score[1],
                             'batch_size': self.batch_size,
                             'epochs': self.epochs,
                             })
