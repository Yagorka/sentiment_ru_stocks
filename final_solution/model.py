from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from final_solution.utils import text_to_text_with_X


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask





#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model_bert = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

class BertClassifier(nn.Module):

    def __init__(self, model_bert, dropout=0.3):

        super(BertClassifier, self).__init__()

        self.bert = model_bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 5)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(512, 5)
        # self.relu2 = nn.ReLU()

    def forward(self, input_id, mask):

        _, out = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)

        return out

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, strs, companies, id_to_words):
        self.strs = strs
        self.companies = companies
        self.id_to_words = id_to_words
        self.empty_str = []

    def __len__(self):
        return sum([len(i) for i in self.companies])
         
    def __iter__(self):
         # yield only valid data, skip Nones
         for i, comp_in_str in enumerate(self.companies):
            if len(comp_in_str)>0:
                for j in comp_in_str:
                    s, num_s, num_comp = text_to_text_with_X(self.strs[i], self.id_to_words[j]), i, j
                    yield tokenizer(s, padding='max_length', truncation=True, max_length=512, return_tensors='pt'), num_s, num_comp
            else:
                self.empty_str.append(i)

def evaluate(model, strs, companies, id_to_words):

    test = MyDataset(strs, companies, id_to_words)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=20)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dict_res = {}
    res = []

    if use_cuda:

        model = model.cuda()

    with torch.no_grad():
    

        for s, num_s, num_comp in test_dataloader:
            

            mask = s['attention_mask'].to(device)
            input_id = s['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            res = output.argmax(dim=1).detach().cpu()
            for n_s, num_comp, sent in zip(num_s.tolist(), num_comp.tolist(), (res+1).tolist()):
                if n_s in dict_res:
                    dict_res[n_s].append((num_comp, float(sent)))
                else:
                    dict_res[n_s] = [(num_comp, float(sent))]
    dict_res['empty'] = test.empty_str
    return dict_res



    