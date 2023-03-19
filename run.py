import torch
import json
import torch.optim as optim
from transformers import LongT5ForConditionalGeneration, T5ForConditionalGeneration, AutoTokenizer
from torchmetrics.text.rouge import ROUGEScore
from data import data_helper
from pprint import pprint


def save_log(batch_idx, predictions, targets, rouge_scores):
    with open("log/preds/prediction_{}".format(batch_idx), "w") as f:
        json.dump(predictions, f)

    with open("log/targets/target_{}".format(batch_idx), "w") as f:
        json.dump(targets, f)

    with open("log/metrics/rouge_score_{}".format(batch_idx), "w") as f:
        json.dump(rouge_scores, f)


def train_epoch(config, epoch, train_loader, val_loader, model, optimizer, metric, tokenizer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        text_input_ids = batch['text_input_ids'].cuda()
        text_attention_mask = batch['text_attention_mask'].cuda()
        label_input_ids = batch['label_input_ids'].cuda()
        # label_attention_mask = batch['label_attention_mask'].cuda()

        output = model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            # decoder_input_ids=label_input_ids,
            # decoder_attention_mask=label_attention_mask
            labels=label_input_ids
        )

        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % config.train_checkpoint == 0:
            print("Epoch: {:3d} | Batch: {:6d} | Loss: {:8.6f} ".format(epoch, batch_idx + 1, loss.item()))

        if (batch_idx + 1) % config.eval_checkpoint == 0:
            predictions, targets, rouge_score = evaluate(val_loader, model, metric, tokenizer)
            pprint(rouge_score)
            save_log(batch_idx, predictions, targets, rouge_score)
            model.train()


def evaluate(val_loader, model, metric, tokenizer):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            text_input_ids = batch['text_input_ids'].cuda()
            text_attention_mask = batch['text_attention_mask'].cuda()
            label_input_ids = batch['label_input_ids'].cuda()

            output = model.generate(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                max_length=512,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            prediction = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in
                          output]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                      label_input_ids]

            predictions.extend(prediction)
            targets.extend(target)

    rouge_score = metric(predictions, targets)

    return predictions, targets, rouge_score


def train(config):

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(config.model_config)
    # model = T5ForConditionalGeneration.from_pretrained(config.model_config).to(device)
    model = LongT5ForConditionalGeneration.from_pretrained(config.model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    metric = ROUGEScore()

    train_loader, val_loader = data_helper(config)

    for epoch in range(config.num_epochs):
        train_epoch(config, epoch, train_loader, val_loader, model, optimizer, metric, tokenizer)
