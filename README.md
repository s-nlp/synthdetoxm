# synthdetoxm

Official implementation of NAACL 2025 Main Conference Paper "Modern LLMs are Few-Shot Parallel Detoxification Data Annotators"


### Instructions

1. Put source non-parallel data into `data/` directory
2. Run `python src/get_data.py`
3. Generate Few Shot demonstrations from the multiparadetox dataset using `python src/get_fewshot.py`
4. Generate a synthetic dataset using `generate_detox_data.sh`
5. Clean the synthetic dataset using `clean_dataset.sh`
6. Train the models using `train_mt0.sh`
7. Run inference of trained models using `inference_of_trained_models.sh`
8. Run final eval using `run_final_eval.sh`
9. Do SBS using `python src/sbs.py`
