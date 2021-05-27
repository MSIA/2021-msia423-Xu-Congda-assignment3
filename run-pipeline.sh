python3 run.py acquire --config=config/config.yaml
python3 run.py load --config=config/config.yaml --output 'data/clouds.csv'
python3 run.py prepare_features --input 'data/clouds.csv' --config=config/config.yaml --output 'data/features.csv'
python3 run.py prepare_additional_features --input 'data/features.csv' --config=config/config.yaml --output 'data/features.csv'
python3 run.py prepare_target --input 'data/clouds.csv' --config=config/config.yaml --output 'data/target.pkl'
python3 run.py split --input 'data/features.csv' 'data/target.pkl' --config=config/config.yaml --output 'data/X_train.csv' 'data/X_test.csv' 'data/y_train.pkl' 'data/y_test.pkl'
python3 run.py train --input 'data/X_train.csv' 'data/y_train.pkl' --config=config/config.yaml --output 'data/rf.sav'
python3 run.py score --input 'data/rf.sav' 'data/X_test.csv' --config=config/config.yaml --output 'data/ypred_prob_test.npy' 'data/ypred_bin_test.npy'
python3 run.py evaluate --input 'data/y_test.pkl' 'data/ypred_prob_test.npy' 'data/ypred_bin_test.npy'
