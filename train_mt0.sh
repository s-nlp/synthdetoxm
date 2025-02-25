#!/bin/bash


python src/train_mt0.py --config_path configs/mt0_xl_config.json
python src/train_mt0.py --config_path configs/mt0_xl_config.json --skip_first 3
python src/train_mt0.py --config_path configs/mt0_xl_config.json --skip_first 6
python src/train_mt0.py --config_path configs/mt0_xl_config.json --skip_first 9
python src/train_mt0.py --config_path configs/mt0_xl_config.json --full_training

