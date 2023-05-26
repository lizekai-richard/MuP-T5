from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stanford_openie import StanfordOpenIE
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from datasets import load_dataset


class TrialDataset(Dataset):

    def __init__(self, mode, tokenizer_config, max_length=512, top_k=20):
        super().__init__()
        if mode == 'train':
            self.data, _ = load_dataset("allenai/mup", split=["train", "validation"])
        else:
            _, self.data = load_dataset("allenai/mup", split=["train", "validation"])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
        self.openie_config = {
            'openie.affinity_probability_cap': 3 / 4,
        }
        self.client = StanfordOpenIE(properties=self.openie_config)
        self.rouge = ROUGEScore()
        self.processed_data = []
        self.top_k = top_k
        self.max_length = max_length
        self._extract_triplets()

    def _extract_triplets_example(self, text):
        triples = []
        for triple in self.client.annotate(text):
            subject_entity = triple['subject']
            object_entity = triple['object']
            relation = triple['relation']
            triples.append((subject_entity, relation, object_entity))
        return triples

    def _extract_triplets(self):
        print("Extracting relational triplets...")
        for example in tqdm(self.data):
            sci_triples = self._extract_triplets_example(example['text'])
            example['sci_claims'] = sci_triples
            self.processed_data.append(example)

    def _select_top_k_triplets(self, example, k):
        scores = []
        triplets = example['sci_claims']
        for i, triple in enumerate(triplets):
            triple_text = ' '.join(triple)
            source_text = example['text']
            score = self.rouge(triple_text, source_text)
            scores.append((i, score['rouge1_fmeasure'], score['rouge2_fmeasure'],
                           score['rougeL_fmeasure']))
        scores.sort(key=lambda x: (x[1] + x[2] + x[3]) / 3, reverse=True)
        top_k_indices = [t[0] for t in scores[:k + 1]]
        top_k_triplets = [triplets[i] for i in top_k_indices]
        return top_k_triplets

    def __getitem__(self, index):
        processed_example = self.processed_data[index]
        triplets = self._select_top_k_triplets(processed_example, self.top_k)

        input_text = "Summarize: " + processed_example['text']
        sci_claim_text = ""
        for triplet in triplets:
            sci_claim_text += ' '.join(triplet)
        summary = processed_example['summary']

        tokenized_text = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors='pt')
        tokenized_sci_claim = self.tokenizer(sci_claim_text, max_length=self.max_length, padding='max_length',
                                             truncation=True, return_tensors='pt')

        tokenized_summary = self.tokenizer(summary, max_length=self.max_length, padding='max_length', truncation=True,
                                           return_tensors='pt')

        text_input_ids = tokenized_text['input_ids']
        text_attention_mask = tokenized_text['attention_mask']
        claim_input_ids = tokenized_sci_claim['input_ids']
        claim_attention_mask = tokenized_sci_claim['attention_mask']
        summary_input_ids = tokenized_summary['input_ids']

        return text_input_ids, text_attention_mask, claim_input_ids, claim_attention_mask, summary_input_ids, summary

    def __len__(self):
        return len(self.data)


class TrialModel(nn.Module):

    def __init__(self, t5_config, t5_hidden_size, hidden_size, num_beams=5):
        super().__init__()
        self.num_beams = num_beams
        self.tokenizer = AutoTokenizer.from_pretrained(t5_config)

        self.text_encoder = T5EncoderModel.from_pretrained(t5_config)
        self.signal_encoder = T5EncoderModel.from_pretrained(t5_config)

        self.text_mapping = nn.Linear(t5_hidden_size, hidden_size)
        self.signal_mapping = nn.Linear(t5_hidden_size, hidden_size)
        self.forget_gate = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_gate = nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.decoder = T5ForConditionalGeneration.from_pretrained(t5_config)

    def _encoder_forward(self, src, src_mask, sig, sig_mask):
        # [bsz, src_len, t5_hdz]
        src_embedding = self.text_encoder(src, src_mask).last_hidden_state
        # [bsz, sig_len, sig_hdz]
        sig_embedding = self.signal_encoder(sig, sig_mask).last_hidden_state
        # [bsz, src_len, hdz]
        src_embedding = F.relu(self.text_mapping(src_embedding))
        # [bsz, sig_len, hdz]
        sig_embedding = F.relu(self.signal_mapping(sig_embedding))
        # [bsz, src_len, sig_len]
        sim_mat = torch.bmm(src_embedding, sig_embedding.permute(0, 2, 1).contiguous())
        attn_mat = F.softmax(sim_mat, dim=-1)
        # [bsz, src_len, hdz]
        updated_src_embedding = torch.bmm(attn_mat, sig_embedding)
        # [bsz, src_len, 2 * hdz]
        concat_embedding = torch.cat([src_embedding, updated_src_embedding], dim=-1)
        forget_gate_output = F.sigmoid(self.forget_gate(concat_embedding))
        update_gate_output = F.tanh(self.update_gate(concat_embedding))
        output = forget_gate_output * update_gate_output
        return output

    def forward(self, src, src_mask, sig, sig_mask, tgt):
        encoder_outputs = self._encoder_forward(src, src_mask, sig, sig_mask)
        decoder_output = self.decoder(encoder_outputs=encoder_outputs, labels=tgt)
        return decoder_output

    def generate(self, src, src_mask):
        generation = self.decoder.generate(
            input_ids=src,
            attention_mask=src_mask,
            max_length=256,
            num_beams=self.num_beams,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in generation]


def logger(log):
    with open("log.txt", "a+") as f:
        f.write(log)


def train_epoch(epoch, model, data_loader, optimizer, device):
    model.train()
    for batch_idx, batch in enumerate(data_loader):
        src, src_mask, sig, sig_mask, summary, _ = batch
        src, src_mask, sig, sig_mask, summary = src.to(device), src_mask.to(device), sig.to(device), \
            sig_mask.to(device), summary.to(device)

        output = model(src, src_mask, sig, sig_mask, summary)

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1000 == 0:
            with torch.no_grad():
                loss = output.loss.item()
                print("Epoch: {:8d}| Batch: {:8d}| Loss: {:8.6f}".format(epoch, batch_idx + 1, loss))
                logger("Epoch: {:8d}| Batch: {:8d}| Loss: {:8.6f}\n".format(epoch, batch_idx + 1, loss))


def evaluate(model, data_loader, device):
    model.eval()
    pred_summaries = []
    gold_summaries = []
    for batch_idx, batch in enumerate(data_loader):
        src, src_mask, summary = batch[0], batch[1], batch[5]
        src, src_mask = src.to(device), src_mask.to(device)
        generated_summary = model.generate(src, src_mask)
        pred_summaries.extend(generated_summary)
        gold_summaries.extend(summary)

    rouge = ROUGEScore()
    score = rouge(pred_summaries, gold_summaries)
    print("Evaluating...")
    print("Rouge1: {:6.6f}| Rouge2: {:6.6f} | RougeL: {:6.6f}".format(score['rouge1_fmeasure'],
                                                                      score['rouge2_fmeasure'],
                                                                      score['rougeL_fmeasure']))
    logger("Rouge1: {:6.6f}| Rouge2: {:6.6f} | RougeL: {:6.6f}\n".format(score['rouge1_fmeasure'],
                                                                      score['rouge2_fmeasure'],
                                                                      score['rougeL_fmeasure']))


def train(num_epochs, model, train_loader, dev_loader, optimizer, device):
    print("Starts training")
    for epoch in range(num_epochs):
        train_epoch(epoch, model, train_loader, optimizer, device)
        evaluate(model, dev_loader, device)


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_ds = TrialDataset("train", "t5-base")
    dev_ds = TrialDataset("dev", "t5-base")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=32)

    model = TrialModel("t5-base", 768, 300).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train(
        num_epochs=5,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        device=device
    )

