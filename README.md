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


### Citation

```
@inproceedings{moskovskiy-etal-2025-synthdetoxm,
    title = "{S}ynth{D}etox{M}: {M}odern {LLM}s are Few-Shot Parallel Detoxification Data Annotators",
    author = "Moskovskiy, Daniil  and
      Sushko, Nikita  and
      Pletenev, Sergey  and
      Tutubalina, Elena  and
      Panchenko, Alexander",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.294/",
    pages = "5714--5733",
    ISBN = "979-8-89176-189-6",
    abstract = "Existing approaches to multilingual text detoxification are hampered by the scarcity of parallel multilingual datasets. In this work, we introduce a pipeline for the generation of multilingual parallel detoxification data. We also introduce SynthDetoxM, a manually collected and synthetically generated multilingual parallel text detoxification dataset comprising 16,000 high-quality detoxification sentence pairs across German, French, Spanish and Russian. The data was sourced from different toxicity evaluation datasets and then rewritten with nine modern open-source LLMs in few-shot setting. Our experiments demonstrate that models trained on the produced synthetic datasets have superior performance to those trained on the human-annotated MultiParaDetox dataset even in data limited setting. Models trained on SynthDetoxM outperform all evaluated LLMs in few-shot setting. We release our dataset and code to help further research in multilingual text detoxification."
}
```
