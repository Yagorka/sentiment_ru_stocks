

def func_id_to_words(id, id_to_words):
    return id_to_words[id]


def text_to_ids(lower_text, loaded_id_to_names):
    all_ids = []
    for k, v in loaded_id_to_names.items():
        for name in v:
            i = lower_text.find(name)
            if i!=-1:
                all_ids.append(k)
    return list(set(all_ids))
def text_to_text_with_X(text, words):
    for i in words:
        text = text.replace(i, 'X')
    # sent_text = text.split('. ')
    # result_text = '. '.join([i for i in sent_text if 'X' in i])
    # if len(result_text)<20:
    #     return text
    return text

    
