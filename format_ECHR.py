from tqdm import tqdm
import os
import json
import re
import pickle
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import LongformerTokenizer

def get_data(src_path='ECHR_v2/train', out_path='ECHR_v2/train_augmented'):
    error_cnt = []
    error_cit = []
    all_citations = []
    got_cit = []
    total = 0
    cit_in_data = False

    paths = ['ECHR_v2/train', 'ECHR_v2/dev', 'ECHR_v2/test']
    for case_path in paths:
        for item in tqdm(os.listdir(case_path)):
            if item.endswith('.json'):
                total += 1
                with open(os.path.join(case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    all_citations.append(data["case_no"])

    for item in tqdm(os.listdir(src_path)):
        if item.endswith('.json'):

            with open(os.path.join(src_path, item), "r") as json_file:
                data = json.load(json_file)

                try:
                    arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
                    arguments = " ".join(arguments.split())
                except:
                    error_cnt.append(data["case_no"])
                    arguments = []

                try:
                    old_cit = list(set(re.findall("\d{3,9}/\d{1,2}", arguments)))

                    cit = []
                    for c in old_cit:
                        if c != data["case_no"]:
                            cit.append(c)

                    if len(cit) == 0:
                        error_cit.append(data["case_no"])

                except:
                    cit = []
                    error_cit.append(data["case_no"])

                for c in cit:
                    if c in all_citations:
                        cit_in_data = True

                got_cit.append(cit_in_data)
                cit_in_data = False

                data["arguments"] = arguments
                data["citations"] = cit

            with open(os.path.join(out_path, item), "w") as out_file:
                json.dump(data, out_file, indent=1)

    print(error_cnt, error_cit, total)

def add_facts(src_path='ECHR_v2/train', out_path='ECHR_v2/train_augmented'):

    cnt_false = 0
    cnt_all = 0

    all_facts = {}

    paths = ['ECHR_v2/train', 'ECHR_v2/dev', 'ECHR_v2/test']
    for case_path in paths:
        for item in tqdm(os.listdir(case_path)):
            if item.endswith('.json'
                             ):
                with open(os.path.join(case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    for d in data["case_no"].split(";"):
                        if len(data["facts"]) != 0:
                            instance_out = data['alleged_violations']
                            instance_facts = data["facts"]
                            if len(instance_out) != 0:
                                for i in instance_out:
                                    instance_facts.insert(0, i)
                            all_facts[d] = instance_facts

    for item in tqdm(os.listdir(src_path)):
        if item.endswith('.json'):
            with open(os.path.join(src_path, item), "r") as json_file:

                cnt_all += 1
                data = json.load(json_file)
                data["precedent_facts"] = [" ".join(data["facts"])]

                cit_added = False
                for cit in data["citations"]:
                    if cit != data["case_no"]:
                        try:
                            data["precedent_facts"].append(" ".join(all_facts[cit]))
                            cit_added = True
                        except:
                            pass

                if cit_added == False:
                    cnt_false += 1
                    data["precedent_facts"] = []

            with open(os.path.join(out_path, item), "w") as out_file:
                json.dump(data, out_file, indent=1)

    print(cnt_false, cnt_all)
    return all_facts

def add_arguments(src_path='ECHR_v2/train', out_path='ECHR_v2/train_augmented'):

    all_facts = {}

    paths = ['ECHR_v2/train_augmented', 'ECHR_v2/dev_augmented', 'ECHR_v2/test_augmented']
    for case_path in paths:
        for item in tqdm(os.listdir(case_path)):
            if item.endswith('.json'):
                with open(os.path.join(case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    for d in data["case_no"].split(";"):
                        if len(data["arguments"]) != 0:

                            instance_out = data['alleged_violations']
                            instance_args = data["arguments"]
                            if len(instance_out) != 0:
                                for i in instance_out:
                                    instance_args = i + " " + instance_args

                            all_facts[d] = instance_args

    for item in tqdm(os.listdir(src_path)):
        if item.endswith('.json'):
            with open(os.path.join(src_path, item), "r") as json_file:
                data = json.load(json_file)
                data["precedent_arguments"] = [" ".join(data["facts"])]

                cit_added = False
                for cit in data["citations"]:
                    if cit != data["case_no"]:
                        try:
                            data["precedent_arguments"].append(all_facts[cit])
                            cit_added = True
                        except:
                            pass

                if cit_added == False:
                    data["precedent_arguments"] = []

            with open(os.path.join(out_path, item), "w") as out_file:
                json.dump(data, out_file, indent=1)

    return all_facts


def add_both(src_path='ECHR_v2/train', out_path='ECHR_v2/train_augmented'):
    all_facts = {}

    paths = ['ECHR_v2/train_augmented', 'ECHR_v2/dev_augmented', 'ECHR_v2/test_augmented']
    for case_path in paths:
        for item in tqdm(os.listdir(case_path)):
            if item.endswith('.json'):
                with open(os.path.join(case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    for d in data["case_no"].split(";"):
                        if len(data["arguments"]) != 0 and len(data["facts"]) != 0:

                            instance_out = data["violated_articles"]
                            instance_args = data["arguments"]
                            if len(instance_out) != 0:
                                for i in instance_out:
                                    instance_args = i + " " + instance_args

                            instance_out = data["violated_articles"]
                            instance_facts = data["facts"]
                            if len(instance_out) != 0:
                                for i in instance_out:
                                    instance_facts.insert(0, i)

                            all_facts[d] = [instance_args, " ".join(instance_facts)]




    for item in tqdm(os.listdir(src_path)):
        if item.endswith('.json'):
            with open(os.path.join(src_path, item), "r") as json_file:
                data = json.load(json_file)
                data["precedent_both"] = [" ".join(data["facts"])]

                cit_added = False
                for cit in data["citations"]:
                    if cit != data["case_no"]:
                        try:
                            data["precedent_both"].append(all_facts[cit][0])
                            data["precedent_both"].append(all_facts[cit][1])
                            cit_added = True
                        except:
                            pass

                if cit_added == False:
                    data["precedent_both"] = []

            with open(os.path.join(out_path, item), "w") as out_file:
                json.dump(data, out_file, indent=1)

    return all_facts


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, tokenizer, max=4096):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # For every sentence...
    input_ids = []
    attention_masks = []

    for sent in tqdm(data):
        sent = sent[:500000] # Speeds the process up for documents with a lot of precedent we would truncate anyway.
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True,
        )

        # Add the outputs to the lists
        input_ids.append([encoded_sent.get('input_ids')])
        attention_masks.append([encoded_sent.get('attention_mask')])

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def get_data_pickled(src_path='ECHR/EN_train', binary=True, seq_len=100, data_type="precedent_facts"):
    all_ids = []
    all_text = []
    all_targets = []
    non_cnt = 0
    tru_cnt = 0
    all_multi_targets = []

    sizes = []

    for item in tqdm(os.listdir(src_path)):
        if item.endswith('.json'):
            with open(os.path.join(src_path, item)) as json_file:
                data = json.load(json_file)

                try:
                    text = " ".join(data[data_type])
                except:
                    text = []

                test = " ".join(data["precedent_facts"])
                if len(test) == 0:
                    text = []

                if len(text) != 0:

                    if data_type == "arguments":
                        text = " ".join(data[data_type].split()[:seq_len])
                    elif data_type == "facts":
                        text = " ".join(" ".join(data[data_type]).split()[:seq_len])
                    else:
                        text = [" ".join(t.split()[:seq_len]) for t in data[data_type]]
                        text = " ".join(text)

                    case_id = str(data["case_no"])
                    all_ids.append(case_id)
                    all_text.append(text)

                    violation = len(data["violated_articles"]) != 0

                    if violation:
                        all_targets.append(1)
                        all_multi_targets.append(data["violated_articles"])
                        non_cnt += 1
                    else:
                        all_targets.append(0)
                        tru_cnt += 1
                        all_multi_targets.append(["0"])

                    sizes.append(len(text.split(" ")))

    # print(tru_cnt, non_cnt)
    if binary:
        y = np.array(all_targets)
    else:
        y = np.array(all_multi_targets)

    return np.array(all_text), y, all_ids

def run_tokenization(binary=True, max_len=4096, seq_len=100, type="facts"):

    train_X, train_y, train_ids = get_data_pickled('ECHR_v2/train_augmented', binary, seq_len, type)
    val_X, val_y, val_ids = get_data_pickled('ECHR_v2/dev_augmented', binary, seq_len, type)
    test_X, test_y, test_ids = get_data_pickled('ECHR_v2/test_augmented', binary, seq_len, type)

    print(f"Train Size:{len(train_X)} | Validation Size:{len(val_X)} | Test Size:{len(test_X)} | Total: {len(train_X)+len(val_X)+len(test_X)}")

    if binary:
        out_dim = 2

    else:
        mlb = MultiLabelBinarizer()
        train_y = mlb.fit_transform(train_y)
        val_y = mlb.transform(val_y)
        test_y = mlb.transform(test_y)

        out_dim = len(train_y[1])

    list(mlb.classes_)

    print(f"Number of Articles to Classify into: {out_dim}")

    # Concatenate train data and test data
    # all_X = np.concatenate([train_X, val_X])

    # Load the longformer tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    # Encode our concatenated data
    # encoded_text = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_X]

    # Find the maximum length
    # data_max_len = max([len(sent) for sent in encoded_text])
    # print('Max length: ', data_max_len)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(train_y)
    val_labels = torch.tensor(val_y)
    test_labels = torch.tensor(test_y)

    # Print sentence 0 and its encoded token ids
    token_ids = list(preprocessing_for_bert([train_X[0]], tokenizer, max=max_len)[0].squeeze().numpy())
    print('Original: ', train_X[0])
    print('Token IDs: ', token_ids)

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(train_X, tokenizer, max=max_len)

    with open("pretokenized/" + type + "/tokenized_train.pkl", "wb") as f:
        pickle.dump([train_inputs, train_masks, train_labels, train_ids], f, protocol=4)

    val_inputs, val_masks = preprocessing_for_bert(val_X, tokenizer, max=max_len)

    with open("pretokenized/" + type + "/tokenized_val.pkl", "wb") as f:
        pickle.dump([val_inputs, val_masks, val_labels, val_ids], f, protocol=4)

    test_inputs, test_masks = preprocessing_for_bert(test_X, tokenizer, max=max_len)

    with open("pretokenized/" + type + "/tokenized_test.pkl", "wb") as f:
        pickle.dump([test_inputs, test_masks, test_labels, test_ids], f, protocol=4)

    print("DONE dump")

def run():

    get_data('ECHR_v2/train', 'ECHR_v2/train_augmented')
    add_arguments('ECHR_v2/train_augmented', 'ECHR_v2/train_augmented')
    add_facts('ECHR_v2/train_augmented', 'ECHR_v2/train_augmented')
    add_both('ECHR_v2/train_augmented', 'ECHR_v2/train_augmented')

    get_data('ECHR_v2/dev', 'ECHR_v2/dev_augmented')
    add_facts('ECHR_v2/dev_augmented', 'ECHR_v2/dev_augmented')
    add_arguments('ECHR_v2/dev_augmented', 'ECHR_v2/dev_augmented')
    add_both('ECHR_v2/dev_augmented', 'ECHR_v2/dev_augmented')

    get_data('ECHR_v2/test', 'ECHR_v2/test_augmented')
    add_facts('ECHR_v2/test_augmented', 'ECHR_v2/test_augmented')
    add_arguments('ECHR_v2/test_augmented', 'ECHR_v2/test_augmented')
    add_both('ECHR_v2/test_augmented', 'ECHR_v2/test_augmented')

    run_tokenization(binary=False, type="precedent_facts", seq_len=512, max_len=3072)
    run_tokenization(binary=False, type="precedent_arguments", seq_len=512, max_len=3072)
    run_tokenization(binary=False, type="precedent_both", seq_len=512, max_len=3072)

    run_tokenization(binary=False, type="facts", seq_len=512, max_len=3072)
    run_tokenization(binary=False, type="arguments", seq_len=512, max_len=3072)


if __name__ == '__main__':
    run()