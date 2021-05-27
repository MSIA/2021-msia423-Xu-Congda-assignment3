import argparse
import logging

import yaml
import pandas as pd
import numpy as np
import pickle

logging.basicConfig(format='%(name)-12s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('assignment3-reproducibility')

import src.acquire_data as acquire
import src.load_data as load
import src.prepare_data as prepare
import src.split_data as split
import src.train_model as train
import src.score_model as score
import src.evaluation as evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model pipeline for clouds classification")

    parser.add_argument('step', help='Which step to run', choices=['acquire', 'load', 'prepare_features', 'prepare_additional_features', 'prepare_target', 'split', 'train', 'score', 'evaluate', 'test'])
    parser.add_argument('--input', '-i', nargs='+', default=None, help='Path to input data')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--output', '-o', nargs='+', default=None, help='Path to save output CSV (optional, default = None)')

    args = parser.parse_args()

    # load configuration file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info("Configuration file loaded from %s" % args.config)

    # load input data from specified path
    if args.input is not None:
        # case when there is only one input path
        if len(args.input) == 1:
            # load a csv file into a pandas dataframe
            input = pd.read_csv(args.input[0])
            logger.info('Input data loaded from %s', args.input)
        # case when there are multiple input paths
        else:
            inputs = []
            for i in args.input:
                # load a csv file into a pandas dataframe
                if i.endswith('.csv'):
                    input = pd.read_csv(i)
                # load a sav file into a ML model
                elif i.endswith('.sav'):
                    with open(i, 'rb') as f:
                        input = pickle.load(f)
                # load a pkl file into a pandas series
                elif i.endswith('.pkl'):
                    input = pd.read_pickle(i)
                # load a npy file into a numpy ndarray
                elif i.endswith('.npy'):
                    with open(i, 'rb') as f:
                        input = np.load(f)
                logger.info('Input data loaded from %s', i)
                inputs.append(input)

    if args.step == 'acquire':
        acquire.acquire_data(**config['acquire_data'])
    elif args.step == 'load':
        output = load.load_data(**config['load_data'])
    elif args.step == 'prepare_features':
        output = prepare.get_features(input, **config['prepare_data']['get_features'])
    elif args.step == 'prepare_additional_features':
        output = prepare.additional_features(input, **config['prepare_data']['additional_features'])
    elif args.step == 'prepare_target':
        output = prepare.get_target(input, **config['prepare_data']['get_target'])
    elif args.step == 'split':
        output1, output2, output3, output4 = split.train_test_split(inputs[0], inputs[1], **config['split_data'])
        output = [output1, output2, output3, output4]
    elif args.step == 'train':
        output = train.train_model(inputs[0], inputs[1], **config['train_model'])
    elif args.step == 'score':
        output1, output2 = score.score_model(inputs[0], inputs[1], **config['score_model'])
        output = [output1, output2]
    elif args.step == 'evaluate':
        evaluate.evaluation(inputs[0], inputs[1], inputs[2])

    # save artifacts to specified output path
    if args.output is not None:
        # case when there is only one output path
        if len(args.output) == 1:
            # save a pandas dataframe to a csv file
            if type(output) == pd.core.frame.DataFrame:
                output.to_csv(args.output[0], index=False)
            # save a pandas series to a pkl file
            elif type(output) == pd.core.series.Series:
                output.to_pickle(args.output[0])
            # save a ML model to a sav file
            else:
                with open(args.output[0], 'wb') as f:
                    pickle.dump(output, f)
            logger.info("Output saved to %s" % args.output[0])
        # case where there are multiple output paths
        else:
            for i in range(len(output)):
                # save a pandas dataframe to a csv file
                if type(output[i]) == pd.core.frame.DataFrame:
                    output[i].to_csv(args.output[i], index=False)
                # save a pandas series to a pkl file
                elif type(output[i]) == pd.core.series.Series:
                    output[i].to_pickle(args.output[i])
                # save a numpy ndarray to a npy file
                else:
                    with open(args.output[i], 'wb') as f:
                        np.save(f, output[i])
                logger.info("Output saved to %s" % args.output[i])