
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def get_data_from_path(path:str):
    # path = "datas/hlm/data.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    PROMPT_TEMP_SINGLE = "请以{who}的身份说一句话"
    PROMPT_TEMP_ANSWER = "请以{who}的身份回复以下对话: {dialogue}"
    outdata = []
    for i, p in enumerate(data):
        dialogue = ""       
        for j, (who, words) in enumerate(p):
            if 0 == j:
                d = {
                    'prompt': PROMPT_TEMP_SINGLE.format(who=who),
                    'reply': words
                }
            else:
                d = {
                    'prompt': PROMPT_TEMP_ANSWER.format(who=who, dialogue=dialogue),
                    'reply': words
                }

            outdata.append(d)
            dialogue += f"{who}:{words}\n"

    outdata = Dataset.from_list(outdata)
    return outdata


        
    
def get_tokenized_dataset_from_path(tokenizer: AutoTokenizer, datapath: str):
    path = datapath
    raw_data = get_data_from_path(path)
    def tokenized_map(item):
        prompt = tokenizer.encode(tokenizer.bos_token + item['prompt'], add_special_tokens=False)
        reply = tokenizer.encode(item['reply'] + tokenizer.eos_token, add_special_tokens=False)
        sample = {
            'input_ids': prompt + reply,
            'attention_mask': [1] * (len(prompt) + len(reply)),
            'labels': [-100] * len(prompt) + reply
        }
        return sample
    dataset = raw_data.map(tokenized_map, remove_columns=list(raw_data.features))

    return dataset

def get_hlm_with_split(dataset_config, tokenizer, split):
    return get_tokenized_dataset_from_path(tokenizer, split)

def get_hlm(dataset_config, tokenizer, split):
    train_path = "datas/hlm/data_train.json"
    eval_path = "datas/hlm/data_eval.json"
    train_dataset = get_tokenized_dataset_from_path(tokenizer, train_path)
    eval_dataset = get_tokenized_dataset_from_path(tokenizer, eval_path)
    return DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })

def main():
    tokenizer_name = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    path = "datas/hlm/data.json"
    print("bos_token:", tokenizer.bos_token)
    print("eos_token:", tokenizer.eos_token)
    print(tokenizer("helloworld"))
    print(tokenizer.encode('helloworld'))

    data = get_tokenized_dataset_from_path(tokenizer, path)
    # print(data['input_ids'])


if '__main__' == __name__:
    main()