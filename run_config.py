import importlib
from pathlib import Path
import sys
import time
import os
import glob
import traceback
import socket
import numpy as np
import pandas as pd
from skopt import Optimizer
import signal
import yaml

from utils.exporters.s3 import S3
from utils.notifier import Notifyer

import evaluation.loader as dl
from builtins import Exception
import pickle
import dill

import random
import gc

# telegram notificaitons
CHAT_ID = os.getenv('CHAT_ID', None)
BOT_TOKEN = os.getenv('TELEGRAM_TOKEN', None)
RUNTIME_THRESHOLD = os.getenv('RUNTIME_THRESHOLD', 100) # how many seconds before the runtime should close
RUNTIME_TICK = os.getenv('RUNTIME_TICK', 5) # how many seconds to wait before checking for new configs


class SessionRec():
    def __init__(self, S3=None, notify_mode="soak") -> None:
        self.S3 = S3
        self.N = Notifyer(mode=notify_mode)
        pass

    def main(self, conf, out=None, s3: S3 = None):
        '''
        Execute experiments for the given configuration path
            --------
            conf: string
                Configuration path. Can be a single file or a folder.
            out: string
                Output folder path for endless run listening for new configurations.
        '''
        print('Checking {}'.format(conf))
        # if TELEGRAM_STATUS:
        #     updater.dispatcher.add_handler( CommandHandler('status', status) )

        file = Path(conf)
        if file.is_file():

            print('Loading file')
            self.N.send_message('processing config ' + conf)
            stream = open(str(file))
            c = yaml.load(stream)
            stream.close()

            try:

                self.run_file(c)
                self.N.send_message('finished config ' + conf)

            except (KeyboardInterrupt, SystemExit):

                self.N.send_message('manually aborted config ' + list[0])
                os.rename(list[0], out + '/' + file.name + str(time.time()) + '.cancled')

                raise

            except Exception:
                print('error for config ', list[0])
                os.rename(list[0], out + '/' + file.name + str(time.time()) + '.error')
                self.N.send_exception('error for config ' + list[0])
                traceback.print_exc()

            exit()

        if file.is_dir():

            if out is not None:
                self.ensure_dir(out + '/out.txt')

                self.N.send_message('waiting for configuration files in ' + conf)
                try: 
                    while True:
                        threshold = 0
                        print('waiting for configuration files in ', conf)

                        list = glob.glob(conf + '/' + '*.yml')
                        if len(list) > 0:
                            try:
                                file = Path(list[0])
                                print('processing config', list[0])
                                self.N.send_message('processing config ' + list[0])

                                stream = open(str(file))
                                c = yaml.load(stream)
                                stream.close()

                                self.run_file(c)

                                print('finished config', list[0])
                                self.N.send_message('finished config ' + list[0])

                                os.rename(list[0], out + '/' + file.name + str(time.time()) + '.done')

                            except (KeyboardInterrupt, SystemExit):

                                self.N.send_message('manually aborted config ' + list[0])
                                os.rename(list[0], out + '/' + file.name + str(time.time()) + '.cancled')
                                while S3.Running:
                                    self.N.send_message(f'Waiting for s3 to finish, trying again in {RUNTIME_TICK*5}s')
                                    time.sleep(RUNTIME_TICK*5)
                                
                                os.kill(os.getpid(), signal.SIGTERM)


                            except Exception:
                                print('error for config ', list[0])
                                os.rename(list[0], out + '/' + file.name + str(time.time()) + '.error')
                                self.N.send_exception('error for config ' + list[0])
                                traceback.print_exc()
                        
                        threshold += RUNTIME_TICK
                        time.sleep(RUNTIME_TICK)
                        if threshold > RUNTIME_THRESHOLD:
                            if S3.Running == False:
                                self.N.send_message(f'no new config files in {RUNTIME_THRESHOLD}s, stopping' + conf)
                                raise SystemExit    
                            else:
                                # If our S3 client arent done sending data, give it another 50s.
                                threshold -= 50
                
                except (KeyboardInterrupt, SystemExit):
                    print('manually aborted')
                    self.N.send_message('manually aborted')
                    os.kill(os.getpid(), signal.SIGTERM)


            else:

                print('processing folder ', conf)

                list = glob.glob(conf + '/' + '*.yml')
                for conf in list:
                    try:

                        print('processing config', conf)
                        self.N.send_message('processing config ' + conf)

                        stream = open(str(Path(conf)))
                        c = yaml.load(stream)
                        stream.close()

                        self.N.run_file(c)

                        print('finished config', conf)
                        self.N.send_message('finished config ' + conf)

                    except (KeyboardInterrupt, SystemExit):
                        self.N.send_message('manually aborted config ' + conf)
                        raise

                    except Exception:
                        print('error for config ', conf)
                        self.N.send_exception('error for config' + conf)
                        traceback.print_exc()

                exit()


    def run_file(self, conf):
        '''
        Execute experiments for one single configuration file
            --------
            conf: dict
                Configuration dictionary
        '''
        if conf['type'] == 'single':
            self.run_single(conf)
        elif conf['type'] == 'window':
            self.run_window(conf)
        elif conf['type'] == 'opt':
            self.run_opt(conf)
        elif conf['type'] == 'bayopt':
            self.run_bayopt(conf)
        else:
            print(conf['type'] + ' not supported')


    def run_single(self, conf, slice=None):
        '''
        Evaluate the algorithms for a single split
            --------
            conf: dict
                Configuration dictionary
            slice: int
                Optional index for the window slice
        '''
        print('run test single')

        algorithms = self.create_algorithms_dict(conf['algorithms'])
        metrics = self.create_metric_list(conf['metrics'])
        evaluation = self.load_evaluation(conf['evaluation'])

        buys = pd.DataFrame()

        if 'type' in conf['data']:
            if conf['data']['type'] == 'hdf':  # hdf5 file
                if 'opts' in conf['data']:
                    # ( path, file, sessions_train=None, sessions_test=None, slice_num=None, train_eval=False )
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'], slice_num=slice,
                                                        **conf['data']['opts'])
                else:
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'], slice_num=slice)

        else:  # csv file (default)
            if 'opts' in conf['data']:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice,
                                                **conf['data']['opts'])
            else:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], slice_num=slice)
            if 'buys' in conf['data'] and 'file_buys' in conf['data']:
                buys = dl.load_buys(conf['data']['folder'], conf['data']['file_buys'])  # load buy actions in addition
        # else:
        #     raise RuntimeError('Unknown data type: {}'.format(conf['data']['type']))

        for m in metrics:
            m.init(train)
            if hasattr(m, 'set_buys'):
                m.set_buys(buys, test)

        results = {}

        for k, a in algorithms.items():
            self.eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, slice=slice, iteration=slice)

        self.print_results(results)
        # TODO: write results to S3
        self.write_results_csv(results, conf, iteration=slice)


    def run_opt_single(self, conf, iteration, globals):
        '''
        Evaluate the algorithms for a single split
            --------
            conf: dict
                Configuration dictionary
            slice: int
                Optional index for the window slice
        '''
        print('run test opt single')

        algorithms = self.create_algorithms_dict(conf['algorithms'])
        for k, a in algorithms.items():
            aclass = type(a)
            if not aclass in globals:
                globals[aclass] = {'key': '', 'best': -1}

        metrics = self.create_metric_list(conf['metrics'])
        metric_opt = self.create_metric(conf['optimize'])
        metrics = metric_opt + metrics
        evaluation = self.load_evaluation(conf['evaluation'])

        train_eval = True
        if 'train_eval' in conf['data']:
            train_eval = conf['data']['train_eval']

        if 'type' in conf['data']:
            if conf['data']['type'] == 'hdf':  # hdf5 file
                if 'opts' in conf['data']:
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'],
                                                        train_eval=train_eval,
                                                        **conf['data'][
                                                            'opts'])  # ( path, file, sessions_train=None, sessions_test=None, slice_num=None, train_eval=False )
                else:
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'],
                                                        train_eval=train_eval)
            # elif conf['data']['type'] == 'csv': # csv file
        else:
            if 'opts' in conf['data']:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval,
                                                **conf['data']['opts'])
            else:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval)

        for m in metrics:
            m.init(train)

        results = {}

        for k, a in algorithms.items():
            self.eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, iteration=iteration, out=False)

        # TODO: write results to S3
        self.write_results_csv(results, conf, iteration=iteration)

        for k, a in algorithms.items():
            aclass = type(a)
            current_value = results[k][0][1]
            if globals[aclass]['best'] < current_value:
                print('found new best configuration')
                print(k)
                print('improvement from {} to {}'.format(globals[aclass]['best'], current_value))
                # 
                self.N.send_message('improvement for {} from {} to {} in test {}'.format(k, globals[aclass]['best'], current_value,
                                                                                iteration))
                globals[aclass]['best'] = current_value
                globals[aclass]['key'] = k

        globals['results'].append(results)

        del algorithms
        del metrics
        del evaluation
        del results
        gc.collect()


    def run_bayopt_single(self, conf, algorithms, iteration, globals):
        '''
        Evaluate the algorithms for a single split
            --------
            conf: dict
                Configuration dictionary
            slice: int
                Optional index for the window slice
        '''
        print('run test opt single')

        for k, a in algorithms.items():
            aclass = type(a)
            if not aclass in globals:
                globals[aclass] = {'key': '', 'best': -1}

        metrics = self.create_metric_list(conf['metrics'])
        metric_opt = self.create_metric(conf['optimize'])
        metrics = metric_opt + metrics
        evaluation = self.load_evaluation(conf['evaluation'])

        train_eval = True
        if 'train_eval' in conf['data']:
            train_eval = conf['data']['train_eval']

        if 'type' in conf['data']:
            if conf['data']['type'] == 'hdf':  # hdf5 file
                if 'opts' in conf['data']:
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'],
                                                        train_eval=train_eval,
                                                        **conf['data'][
                                                            'opts'])  # ( path, file, sessions_train=None, sessions_test=None, slice_num=None, train_eval=False )
                else:
                    train, test = dl.load_data_session_hdf(conf['data']['folder'], conf['data']['prefix'],
                                                        train_eval=train_eval)
            # elif conf['data']['type'] == 'csv': # csv file
        else:
            if 'opts' in conf['data']:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval,
                                                **conf['data']['opts'])
            else:
                train, test = dl.load_data_session(conf['data']['folder'], conf['data']['prefix'], train_eval=train_eval)

        for m in metrics:
            m.init(train)

        results = {}

        for k, a in algorithms.items():
            self.eval_algorithm(train, test, k, a, evaluation, metrics, results, conf, iteration=iteration, out=False)

        #TODO: write results to S3
        self.write_results_csv(results, conf, iteration=iteration)

        for k, a in algorithms.items():
            aclass = type(a)
            current_value = results[k][0][1]
            if globals[aclass]['best'] < current_value:
                print('found new best configuration')
                print(k)
                print('improvement from {} to {}'.format(globals[aclass]['best'], current_value))
                self.N.send_message('improvement for {} from {} to {} in test {}'.format(k, globals[aclass]['best'], current_value,
                                                                                iteration))
                globals[aclass]['best'] = current_value
                globals[aclass]['key'] = k

            globals['current'] = current_value

        globals['results'].append(results)

        del algorithms
        del metrics
        del evaluation
        del results
        gc.collect()


    def run_window(self, conf):
        '''
        Evaluate the algorithms for all slices
            --------
            conf: dict
                Configuration dictionary
        '''

        print('run test window')

        slices = conf['data']['slices']
        slices = list(range(slices))
        if 'skip' in conf['data']:
            for i in conf['data']['skip']:
                slices.remove(i)

        for i in slices:
            print('start run for slice ', str(i))
            self.N.send_message('start run for slice ' + str(i))
            self.run_single(conf, slice=i)


    def run_opt(self, conf):
        '''
        Perform an optmization for the algorithms
            --------
            conf: dict
                Configuration dictionary
        '''

        iterations = conf['optimize']['iterations'] if 'optimize' in conf and 'iterations' in conf['optimize'] else 100
        start = conf['optimize']['iterations_skip'] if 'optimize' in conf and 'iterations_skip' in conf['optimize'] else 0
        print('run opt with {} iterations starting at {}'.format(iterations, start))

        globals = {}
        globals['results'] = []

        for i in range(start, iterations):
            print('start random test ', str(i))
            self.run_opt_single(conf, i, globals)

        global_results = {}
        for results in globals['results']:
            for key, value in results.items():
                global_results[key] = value
        
        #TODO: write results to S3
        self.write_results_csv(global_results, conf)


    def run_bayopt(self, conf):
        '''
        Perform a bayesian optmization for the algorithms using
            --------
            conf: dict
                Configuration dictionary
        '''

        iterations = conf['optimize']['iterations'] if 'optimize' in conf and 'iterations' in conf['optimize'] else 100
        start = conf['optimize']['iterations_skip'] if 'optimize' in conf and 'iterations_skip' in conf['optimize'] else 0
        print('run opt with {} iterations starting at {}'.format(iterations, start))

        globals = {}
        globals['results'] = []

        for entry in conf['algorithms']:

            space_dict = self.generate_space(entry)

            # generate space for algorithm
            opt = Optimizer([values for k, values in space_dict.items()], n_initial_points=conf['optimize']['initial_points'] if 'optimize' in conf and 'initial_points' in conf['optimize'] else 10)

            for i in range(start, iterations):
                print('start bayesian test ', str(i))
                suggested = opt.ask()
                params = { k:v for k,v in zip( space_dict.keys(), suggested ) }

                algo_instance = self.create_algorithm_dict( entry, params )

                self.run_bayopt_single(conf, algo_instance, i, globals)
                res = globals['current']
                opt.tell(suggested, -1 * res)

        global_results = {}
        for results in globals['results']:
            for key, value in results.items():
                global_results[key] = value

        #TODO: write results to S3
        self.write_results_csv(global_results, conf)


    def eval_algorithm(self, train, test, key, algorithm, eval, metrics, results, conf, slice=None, iteration=None, out=True):
        '''
        Evaluate one single algorithm
            --------
            train : Dataframe
                Training data
            test: Dataframe
                Test set
            key: string
                The automatically created key string for the algorithm
            algorithm: algorithm object
                Just the algorithm object, e.g., ContextKNN
            eval: module
                The module for evaluation, e.g., evaluation.evaluation_last
            metrics: list of Metric
                Optional string to add to the file name
            results: dict
                Result dictionary
            conf: dict
                Configuration dictionary
            slice: int
                Optional index for the window slice
        '''
        ts = time.time()
        print('fit ', key)
        self.N.send_message( 'training algorithm ' + key )

        if hasattr(algorithm, 'init'):
            algorithm.init(train, test, slice=slice)

        for m in metrics:
            if hasattr(m, 'start'):
                m.start(algorithm)

        algorithm.fit(train, test)
        print(key, ' time: ', (time.time() - ts))

        if 'results' in conf and 'pickle_models' in conf['results']:
            try:
                self.save_model(key, algorithm, conf)
            except Exception:
                print('could not save model for ' + key)

        for m in metrics:
            if hasattr(m, 'start'):
                m.stop(algorithm)

        results[key] = eval.evaluate_sessions(algorithm, metrics, test, train)
        if out:
            #TODO: write results to S3
            self.N.send_results(message=f"Results for {key} in test {iteration}:\n{results[key]}", results=results[key])
            self.write_results_csv({key: results[key]}, conf, extra=key, iteration=iteration)

        # send_message( 'algorithm ' + key + ' finished ' + ( 'for slice ' + str(slice) if slice is not None else '' ) )

        algorithm.clear()


    def write_results_csv(self, results, conf, iteration=None, extra=None):
        '''
        Write the result array to a csv file, if a result folder is defined in the configuration
            --------
            results : dict
                Dictionary of all results res[algorithm_key][metric_key]
            iteration; int
                Optional for the window mode
            extra: string
                Optional string to add to the file name
        '''
        # TODO write to S3 and put me in exporter class
        if 'results' in conf and 'folder' in conf['results']:
            export_csv = conf['results']['folder'] + 'test_' + conf['type'] + '_' + conf['key'] + '_' + conf['data']['name']
            # if extra is not None:
            #     export_csv += '.' + str(extra)
            if iteration is not None:
                export_csv += '.' + str(iteration)
            export_csv += '.csv'

            self.ensure_dir(export_csv)

            file = open(export_csv, 'w+')
            file.write('Metrics;')

            for k, l in results.items():
                for e in l:
                    file.write(e[0])
                    file.write(';')
                break

            file.write('\n')

            for k, l in results.items():
                file.write(k)
                file.write(';')
                for e in l:
                    file.write(str(e[1]))
                    file.write(';')
                    if len(e) > 2:
                        if type(e[2]) == pd.DataFrame:
                            name = export_csv.replace('.csv', '-') + e[0].replace(':', '').replace(' ', '') + '.csv'
                            e[2].to_csv(name, sep=";", index=False)
                file.write('\n')



    def save_model(self, key, algorithm, conf):
        '''
        Save the model object for reuse with FileModel
            --------
            algorithm : object
                Dictionary of all results res[algorithm_key][metric_key]
            conf : object
                Configuration dictionary, has to include results.pickel_models
        '''

        file_name = conf['results']['folder'] + '/' + conf['key'] + '_' + conf['data']['name'] + '_' + key + '.pkl'
        file_name = Path(file_name)
        self.ensure_dir(file_name)
        file = open(file_name, 'wb')

        # pickle.dump(algorithm, file)
        dill.dump(algorithm, file)

        file.close()


    def print_results(self, res):
        '''
        Print the result array
            --------
            res : dict
                Dictionary of all results res[algorithm_key][metric_key]
        '''
        for k, l in res.items():
            for e in l:
                print(k, ':', e[0], ' ', e[1])


    def load_evaluation(self, module):
        '''
        Load the evaluation module
            --------
            module : string
                Just the last part of the path, e.g., evaluation_last
        '''
        return importlib.import_module('evaluation.' + module)


    def create_algorithms_dict(self, list):
        '''
        Create algorithm instances from the list of algorithms in the configuration
            --------
            list : list of dicts
                Dicts represent a single algorithm with class, a key, and optionally a param dict
        '''

        algorithms = {}
        for algorithm in list:
            Class = self.load_class('algorithms.' + algorithm['class'])

            default_params = algorithm['params'] if 'params' in algorithm else {}
            random_params = self.generate_random_params(algorithm)
            params = {**default_params, **random_params}
            del default_params, random_params

            if 'params' in algorithm:
                if 'algorithms' in algorithm['params']:
                    hybrid_algorithms = self.create_algorithms_dict(algorithm['params']['algorithms'])
                    params['algorithms'] = []
                    a_keys = []
                    for k, a in hybrid_algorithms.items():
                        params['algorithms'].append(a)
                        a_keys.append(k)

            # instance = Class( **params )
            key = algorithm['key'] if 'key' in algorithm else algorithm['class']
            if 'params' in algorithm:
                if 'algorithms' in algorithm['params']:
                    for k, val in params.items():
                        if k == 'algorithms':
                            for pKey in a_keys:
                                key += '-' + pKey
                        elif k == 'file':
                            key += ''
                        else:
                            key += '-' + str(k) + "=" + str(val)
                            key = key.replace(',', '_')

                else:
                    for k, val in params.items():
                        if k != 'file':
                            key += '-' + str(k) + "=" + str(val)
                            key = key.replace(',', '_')
                        # key += '-' + '-'.join( map( lambda x: str(x[0])+'='+str(x[1]), params.items() ) )

            if 'params_var' in algorithm:
                for k, var in algorithm['params_var'].items():
                    for val in var:
                        params[k] = val  # params.update({k: val})
                        kv = k
                        for v in val:
                            kv += '-' + str(v)
                        instance = Class(**params)
                        algorithms[key + kv] = instance
            else:
                instance = Class(**params)
                algorithms[key] = instance

        return algorithms


    def create_algorithm_dict(self, entry, additional_params={}):
        '''
        Create algorithm instance from a single algorithms entry in the configuration with additional params
            --------
            entry : dict
                Dict represent a single algorithm with class, a key, and optionally a param dict
        '''

        algorithms = {}
        algorithm = entry

        Class = self.load_class('algorithms.' + algorithm['class'])

        default_params = algorithm['params'] if 'params' in algorithm else {}

        params = {**default_params, **additional_params}
        del default_params

        if 'params' in algorithm:
            if 'algorithms' in algorithm['params']:
                hybrid_algorithms = self.create_algorithms_dict(algorithm['params']['algorithms'])
                params['algorithms'] = []
                a_keys = []
                for k, a in hybrid_algorithms.items():
                    params['algorithms'].append(a)
                    a_keys.append(k)

        # instance = Class( **params )
        key = algorithm['key'] if 'key' in algorithm else algorithm['class']
        if 'params' in algorithm:
            if 'algorithms' in algorithm['params']:
                for k, val in params.items():
                    if k == 'algorithms':
                        for pKey in a_keys:
                            key += '-' + pKey
                    elif k == 'file':
                        key += ''
                    else:
                        key += '-' + str(k) + "=" + str(val)
                        key = key.replace(',', '_')

            else:
                for k, val in params.items():
                    if k != 'file':
                        key += '-' + str(k) + "=" + str(val)
                        key = key.replace(',', '_')
                    # key += '-' + '-'.join( map( lambda x: str(x[0])+'='+str(x[1]), params.items() ) )

        if 'params_var' in algorithm:
            for k, var in algorithm['params_var'].items():
                for val in var:
                    params[k] = val  # params.update({k: val})
                    kv = k
                    for v in val:
                        kv += '-' + str(v)
                    instance = Class(**params)
                    algorithms[key + kv] = instance
        else:
            instance = Class(**params)
            algorithms[key] = instance

        return algorithms


    def generate_random_params(self, algorithm):
        params = {}

        if 'params_opt' in algorithm:
            for key, value in algorithm['params_opt'].items():
                space = []
                if type(value) == list:
                    for entry in value:
                        if type(entry) == list:
                            space += entry
                            # space.append(entry)
                        elif type(entry) == dict:  # range
                            space += list(self.create_linspace(entry))
                        else:
                            space += [entry]
                            # space += entry
                    chosen = random.choice(space)
                elif type(value) == dict:  # range
                    if 'space' in value:
                        if value['space'] == 'weight':
                            space.append(self.create_weightspace(value))  # {from: 0.0, to: 0.9, in: 10, type: float}
                        elif value['space'] == 'recLen':
                            space.append(self.create_linspace(value))
                    else:
                        space = self.create_linspace(value)  # {from: 0.0, to: 0.9, in: 10, type: float}
                    chosen = random.choice(space)
                    chosen = float(chosen) if 'type' in value and value['type'] == 'float' else chosen
                else:
                    print('not the right type')

                params[key] = chosen

        return params


    def generate_space(self, algorithm):
        params = {}

        if 'params_opt' in algorithm:
            for key, value in algorithm['params_opt'].items():
                if type(value) == list:
                    space = []
                    for entry in value:
                        if type(entry) == list:
                            space += entry
                            # space.append(entry)
                        elif type(entry) == dict:  # range
                            space += list(self.create_linspace(entry))
                        else:
                            space += [entry]
                            # space += entry
                elif type(value) == dict:  # range
                    if 'space' in value:
                        if value['space'] == 'weight':
                            space = []
                            space.append(self.create_weightspace(value))  # {from: 0.0, to: 0.9, in: 10, type: float}
                        elif value['space'] == 'recLen':
                            space = []
                            space.append(self.create_linspace(value))
                    else:
                        if value['type'] == 'float':
                            space = (float(value['from']), float(value['to']))
                        else:
                            space = (int(value['from']), int(value['to']))
                else:
                    print('not the right type')

                params[key] = space

        return params


    def create_weightspace(self, value):
        num = value['num']
        space = []
        sum = 0
        rand = 1
        for i in range(num - 1):  # all weights excluding the last one
            while (sum + rand) >= 1:
                # rand = np.linspace(0, 1, num=0.05).astype('float32')
                rand = round(np.random.rand(), 2)
            space.append(rand)
            sum += rand
            rand = 1

        space.append(round(1 - sum, 2))  # last weight
        return space


    def create_linspace(self, value):
        start = value['from']
        end = value['to']
        steps = value['in']
        space = np.linspace(start, end, num=steps).astype(value['type'] if 'type' in value else 'float32')
        return space


    def create_metric_list(self, list):
        '''
        Create metric class instances from the list of metrics in the configuration
            --------
            list : list of dicts
                Dicts represent a single metric with class and optionally the list length
        '''
        metrics = []
        for metric in list:
            metrics += self.create_metric(metric)

        return metrics


    def create_metric(self, metric):
        metrics = []
        Class = self.load_class('evaluation.metrics.' + metric['class'])
        if 'length' in metric:
            for list_length in metric['length']:
                metrics.append(Class(list_length))
        else:
            metrics.append(Class())
        return metrics


    def load_class(self, path):
        '''
        Load a class from the path in the configuration
            --------
            path : dict of dicts
                Path to the class, e.g., algorithms.knn.cknn.ContextKNNN
        '''
        module_name, class_name = path.rsplit('.', 1)

        Class = getattr(importlib.import_module(module_name), class_name)
        return Class


    def ensure_dir(self, file_path):
        '''
        Create all directories in the file_path if non-existent.
            --------
            file_path : string
                Path to the a file
        '''
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


# if __name__ == '__main__':

#     if len(sys.argv) > 1:
#         main(sys.argv[1], out=sys.argv[2] if len(sys.argv) > 2 else None)
#     else:
#         print('File or folder expected.')


