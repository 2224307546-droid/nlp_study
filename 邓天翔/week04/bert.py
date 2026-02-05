import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, List
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=14)

model.load_state_dict(torch.load("./results/bert.pt"))
model.to(device)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    # 获取第idx个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:  # type: ignore[type-arg]
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        request_text = [request_text]  # type: ignore[assignment]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")

    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=128)
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    pred = []
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = output[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())

    classify_name = ['news_agriculture',
                       'news_game', 'news_house',
                       'news_tech', 'news_military',
                       'news_finance', 'news_world',
                       'news_sports', 'news_car',
                       'news_culture', 'news_travel',
                       'news_edu', 'stock',
                       'news_entertainment']
    classify_result = [classify_name[i] for i in pred]
    return classify_result

text = ["谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分"]
result = model_for_bert(text)
print("预测结果", result)


