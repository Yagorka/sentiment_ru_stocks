import typing as tp
import torch

from final_solution.utils import func_id_to_words, text_to_ids, text_to_text_with_X 
from final_solution.model import model_bert, BertClassifier, evaluate


EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]


def score_texts(
    messages: tp.Iterable[str], loaded_id_to_names: dict, model, *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    str_inp = False
    if type(messages) == str:
        messages = [messages]
        str_inp = True

    companies =  [text_to_ids(i, loaded_id_to_names) for i in messages]
    result = []

    # model = BertClassifier(model_bert)
    # model.load_state_dict(torch.load(weight_dir_path))
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # model = model.to(device)
    res = evaluate(model, messages, companies, loaded_id_to_names)
    empty_index = res['empty']
    for i in range(len(messages)):
        if i in empty_index:
            result.append([tuple()])
        else:
            result.append(res[i])             
    return result[0] if str_inp else result
    
