import json
import pathlib
import typing as tp

import final_solution
from huggingface_hub import hf_hub_download

import time
import os
import pickle
from final_solution.model import BertClassifier, model_bert, torch


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def main():
    texts = load_data()
    texts = [i.lower() for i in texts]
    if os.path.exists(os.path.join('weight_dir', "best_model_state.bin")):
        weight_dir_path = os.path.join('weight_dir', "best_model_state.bin")
    else:
        weight_dir_path = hf_hub_download(repo_id="Yagorka/sentiment_finance_ru", filename="best_model_state.bin", local_dir='weight_dir')
    with open(os.path.join('data', 'saved_words.pkl'), 'rb') as f:
        loaded_id_to_names = pickle.load(f)

    model = BertClassifier(model_bert)
    model.load_state_dict(torch.load(weight_dir_path))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)   
    start = time.time()
    scores = final_solution.solution.score_texts(texts, loaded_id_to_names, model)
    print(time.time()-start, 'секунд')
    save_data(scores)
    


if __name__ == '__main__':
    main()
