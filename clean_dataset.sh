#!/bin/bash

python src/classify_refusals.py
python src/filter_non_detoxifiable_examples.py --toxic_data_path data/toxic_clean_ru.txt --detox_data_path data/predictions_for_paper/de
python src/filter_non_detoxifiable_examples.py --toxic_data_path data/toxic_clean_fr.txt --detox_data_path data/predictions_for_paper/fr
python src/filter_non_detoxifiable_examples.py --toxic_data_path data/toxic_clean_ru.txt --detox_data_path data/predictions_for_paper/ru
python src/filter_non_detoxifiable_examples.py --toxic_data_path data/toxic_clean_es.txt --detox_data_path data/predictions_for_paper/es
python src/get_final_data.py --final_folder_path data/predictions_for_paper/ --calculate_statistics_by_model
python src/create_data_for_training.py
