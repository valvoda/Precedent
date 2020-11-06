import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import LongformerTokenizer, LongformerModel

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MultiLabelBinarizer

import random
import time
import json
import numpy as np
import os
import re
import argparse
import uuid

from tqdm import tqdm
import csv

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, out_dim=2, freeze_bert=False, dropout=0.2, n_hidden=50):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, n_hidden, out_dim

        # Instantiate BERT model
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True, gradient_checkpointing=True)

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask, global_attention):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT

        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


class Classifier:

    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            # print('Device name:', torch.cuda.get_device_name(0))
            # print('Device name:', torch.cuda.get_device_name(1))

        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def text_preprocessing(self, text):
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
    def preprocessing_for_bert(self, data, tokenizer, max=4096):
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
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = tokenizer.encode_plus(
                text=self.text_preprocessing(sent),  # Preprocess sentence
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

    def initialize_model(self, out_dim=2, epochs=4, train_dataloader=None, learning_rate=3e-5, dropout=0.2, n_hidden=50):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        model = BertClassifier(out_dim, dropout=dropout, n_hidden=n_hidden)

        # MULTI GPU support:
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        # Tell PyTorch to run the model on GPU
        model.to(self.device)

        # Create the optimizer
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate, # lr=5e-5,    # Default learning rate
                          eps=1e-8    # Default epsilon value
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
        return model, optimizer, scheduler


    def set_seed(self, seed_value=42):
        """Set seed for reproducibility.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


    def train(self, model, train_dataloader, val_dataloader=None, test_dataloader=None, epochs=4, evaluation=False, loss_fn=False, optimizer=False, scheduler=False, binary=False):
        """Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n")

        unique_id = "models/" + uuid.uuid4().hex + '.pt'
        print(unique_id)
        stop_cnt = 0
        best_loss = 100
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Val F1':^9} | {'Val Prec':^9} | {'Val Rec':^9} | {'Elapsed':^9}")
            print("-" *105)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.

                b_input_ids = b_input_ids.squeeze()
                b_attn_mask = b_attn_mask.squeeze()

                global_attention_mask = torch.zeros(b_input_ids.shape, dtype=torch.long, device=self.device)
                global_attention_mask[:, [0]] = 1

                logits = model(input_ids=b_input_ids, attention_mask=b_attn_mask, global_attention=global_attention_mask)
                # Compute loss and accumulate the loss values

                if binary:
                    loss_each = loss_fn(logits, b_labels)
                else:
                    loss_each = loss_fn(logits, b_labels.float())

                loss = torch.mean(loss_each)
                batch_loss += loss.detach().item()
                total_loss += loss.detach().item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 105)
            # =======================================
            #               Evaluation
            # =======================================
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, val_f1_micro, val_prec_micro, val_rec_micro, _, _, _ = self.evaluate(model, val_dataloader, loss_fn, binary)

                if val_loss < best_loss:
                    best_loss = val_loss
                    stop_cnt = 0
                    torch.save(model, unique_id)
                else:
                    stop_cnt += 1
                    print(f"No Improvement! Stop cnt {stop_cnt}")
                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | "
                    f"{val_f1_micro:^9.2f} | {val_prec_micro:^9.2f} | {val_rec_micro:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 105)
            print("\n")

            if stop_cnt == 1:
                print(f"Early Stopping at {stop_cnt}")
                model = torch.load(unique_id)
                break

        print("Training complete!")

        val_loss, val_accuracy, val_f1_micro, val_prec_micro, val_rec_micro, all_val_losses, _, _ = self.evaluate(model, val_dataloader, loss_fn, binary)
        print(f"Best val loss: {best_loss}, recorded val loss: {val_loss}")
        test_loss, test_accuracy, test_f1_micro, test_prec_micro, test_rec_micro, all_test_losses, all_preds, all_truths = self.evaluate(model, test_dataloader, loss_fn, binary)

        # Print performance over the entire training data
        print(
            f"{'Test Loss':^12} | {'Acc':^9} | {'F1':^9} | {'Precission':^9} | {'Recall':^9}")
        print("-" * 40)
        print(f"{test_loss:^12.6f} | {test_accuracy:^9.2f} | {test_f1_micro:^9.2f} | {test_prec_micro:^9.2f} | {test_rec_micro:^9.2f}")

        return val_loss, val_prec_micro, val_rec_micro, val_f1_micro, test_loss, test_prec_micro, test_rec_micro, test_f1_micro, unique_id, all_test_losses, all_preds, all_truths


    def evaluate(self, model, val_dataloader, loss_fn, binary):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        all_labels = []
        all_preds = []

        all_loss = []
        all_truths = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                b_input_ids = b_input_ids.squeeze()
                b_attn_mask = b_attn_mask.squeeze()

                global_attention_mask = torch.zeros(b_input_ids.shape, dtype=torch.long, device=self.device)
                global_attention_mask[:, [0]] = 1

                logits = model(input_ids=b_input_ids, attention_mask=b_attn_mask, global_attention=global_attention_mask)

            # Compute loss
            if binary:
                loss_each = loss_fn(logits, b_labels)
            else:
                loss_each = loss_fn(logits, b_labels.float())

            all_loss += loss_each.detach().tolist()

            loss = torch.mean(loss_each)
            val_loss.append(loss.item())

            # Get the predictions
            if binary:
                preds = torch.argmax(logits, dim=1).flatten()
                # Calculate the accuracy rate
                accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            else:
                preds = torch.round(torch.sigmoid(logits))
                # Calculate the accuracy rate
                accuracy = (preds == b_labels.float()).cpu().numpy().mean() * 100

            val_accuracy.append(accuracy)

            all_truths += b_labels.float().tolist()

            if binary:
                all_labels += b_labels.cpu().tolist()
                all_preds += preds.cpu().tolist()
            else:
                all_labels += b_labels.cpu().float().tolist()
                all_preds += preds.cpu().float().tolist()

        # Compute the average accuracy and loss over the validation set.
        avg_val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        if binary:
            val_f1 = f1_score(all_labels, all_preds, average="macro") * 100
            val_prec = precision_score(all_labels, all_preds, average="macro") * 100
            val_rec = recall_score(all_labels, all_preds, average="macro") * 100
        else:
            val_f1 = f1_score(all_labels, all_preds, average="micro") * 100
            val_prec = precision_score(all_labels, all_preds, average="micro") * 100
            val_rec = recall_score(all_labels, all_preds, average="micro") * 100

        return avg_val_loss, val_accuracy, val_f1, val_prec, val_rec, all_loss, all_preds, all_truths

    def get_data(self, src_path='ECHR/EN_train', binary=True, seq_len=100):

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

                    text = " ".join(data["precedent_facts"])

                    if len(text) != 0:
                        text = [" ".join(t.split()[:seq_len]) for t in data["precedent_facts"]]
                        text = " ".join(text)

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

        if binary:
            y = np.array(all_targets)
        else:
            y = np.array(all_multi_targets)

        return np.array(all_text), y

    def run(self, epochs=2, binary=True, batch_size=16, max_len=4096, lr=3e-5, dropout=0.2, n_hidden=50, seq_len=100, data_type="precedent_facts"):

        with open("pretokenized/"+data_type+"/tokenized_train.pkl", "rb") as f:
            train_inputs, train_masks, train_labels, train_ids = pickle.load(f)

        with open("pretokenized/"+data_type+"/tokenized_val.pkl", "rb") as f:
            val_inputs, val_masks, val_labels, val_ids = pickle.load(f)

        with open("pretokenized/"+data_type+"/tokenized_test.pkl", "rb") as f:
            test_inputs, test_masks, test_labels, test_ids = pickle.load(f)


        if data_type == "facts" or data_type == "arguments":
            train_inputs, train_masks = train_inputs[:, :, :512], train_masks[:, :, :512]
            val_inputs, val_masks = val_inputs[:, :, :512], val_masks[:, :, :512]
            test_inputs, test_masks = test_inputs[:, :, :512], test_masks[:, :, :512]
        elif data_type == "precedent_both":
            train_inputs, train_masks = train_inputs[:, :, :2560], train_masks[:, :, :2560]
            val_inputs, val_masks = val_inputs[:, :, :2560], val_masks[:, :, :2560]
            test_inputs, test_masks = test_inputs[:, :, :2560], test_masks[:, :, :2560]
        else:
            train_inputs, train_masks = train_inputs[:, :, :1024], train_masks[:, :, :1024]
            val_inputs, val_masks = val_inputs[:, :, :1024], val_masks[:, :, :1024]
            test_inputs, test_masks = test_inputs[:, :, :1024], test_masks[:, :, :1024]

        out_dim = len(train_labels[1])
        print("Classifying into: ", out_dim)
        print("DONE Loading")

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        # Create the DataLoader for our training set
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        # Specify loss function
        if binary:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # self.set_seed(42)    # Set seed for reproducibility
        classifier, optimizer, scheduler = self.initialize_model(out_dim=out_dim, epochs=epochs, train_dataloader=train_dataloader, learning_rate=lr, dropout=dropout, n_hidden=n_hidden)
        val_loss, val_precission, val_recall, val_f1, test_loss, test_precission, test_recall, test_f1, model_name, all_test_losses, all_test_preds, all_test_truths = self.train(classifier, train_dataloader, val_dataloader,
                                                                                                                                                test_dataloader, epochs=epochs, evaluation=True,
                                                                                                                                                 loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                                                                                                                                                 binary=binary)

        # val_loss, val_precission, val_recall, val_f1, test_loss, test_precission, test_recall, test_f1, model_name, all_test_losses = 1, 1, 1, 1, 1, 1, 1, 1, "test", [1, 2, 3, 4, 5]

        self.log_saver(val_precission, val_recall, val_f1, test_precission, test_recall, test_f1, model_name, data_type, dropout, lr, batch_size, n_hidden, val_loss, test_loss, all_test_losses, all_test_preds,  all_test_truths, test_ids)



        return val_loss, val_precission, val_recall, val_f1, test_precission, test_recall, test_f1, model_name


    def log_saver(self, dev_precission, dev_recall, dev_f1, test_precission, test_recall, test_f1, model_name, model_type, dropout, lr, batch_size, n_hidden, dev_loss, test_loss, all_test_losses, all_test_preds,  all_test_truths, test_ids):


        with open("models/" + model_type + "_preds.csv", 'w') as loss_file:
            writer = csv.writer(loss_file)
            for t_id, loss, pred, truth in zip(test_ids, all_test_losses, all_test_preds, all_test_truths):
                writer.writerow([t_id, loss, pred, truth])

        csv_columns = ["model_name", "model_type", "dropout", "lr", "batch_size", "n_hidden", "dev_precission", "dev_recall", "dev_f1", "test_precission", "test_recall", "test_f1", "dev_loss", "test_loss"]
        new_dict_data = [
            {"model_name": model_name, "model_type": model_type, "dropout": dropout, "lr": lr, "batch_size": batch_size, "n_hidden": n_hidden, "dev_precission": dev_precission, "dev_recall": dev_recall, "dev_f1": dev_f1, "test_precission": test_precission, "test_recall":test_recall, "test_f1":test_f1, "dev_loss":dev_loss, "test_loss":test_loss}
        ]

        csv_file = "models/" + model_type + "_results.csv"

        if os.path.isfile(csv_file):
            try:
                with open(csv_file, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    for data in new_dict_data:
                        writer.writerow(data)

            except IOError:
                print("I/O error")

        else:
            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in new_dict_data:
                        writer.writerow(data)

            except IOError:
                print("I/O error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", type=int, default=512, required=False)
    parser.add_argument("--max_length", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--dropout", type=float, default=0.2, required=False)
    parser.add_argument("--n_hidden", type=float, default=50, required=False)
    parser.add_argument("--data_type", type=str, default="precedent_arguments", required=False)
    parser.add_argument("--bin", dest='bin', action='store_true')

    args = parser.parse_args()
    print(args)

    cl = Classifier()
    cl.run(epochs=10, binary=args.bin, max_len=args.max_length, batch_size=args.batch_size,
           lr=args.learning_rate, dropout=args.dropout, n_hidden=args.n_hidden, seq_len=args.seq_length, data_type=args.data_type)
