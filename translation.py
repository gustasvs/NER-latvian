from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
import itertools
import torch
import numpy as np

ner_tag_arr = ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC']

model_name = "Helsinki-NLP/opus-mt-tc-big-en-lv"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

alignment_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
alignment_model = AutoModel.from_pretrained("bert-base-multilingual-cased")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
alignment_model.to(device)

dataset = load_dataset("conll2003",trust_remote_code=True)

def test_alignments(source, translation):
    # pre-processing
    sent_src, sent_tgt = source.strip().split(), translation.strip().split()
    token_src, token_tgt = [alignment_tokenizer.tokenize(word) for word in sent_src], [alignment_tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [alignment_tokenizer.convert_tokens_to_ids(x) for x in token_src], [alignment_tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = alignment_tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=alignment_tokenizer.model_max_length, truncation=True)['input_ids'].to(device), alignment_tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=alignment_tokenizer.model_max_length)['input_ids'].to(device)
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    alignment_model.eval()
    with torch.no_grad():
        out_src = alignment_model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = alignment_model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

    return sent_src, sent_tgt, align_words

def aggregate_embeddings(hidden_states, word_ids):
    embeddings = []
    current_word = None
    current_embs = []
    for i, wid in enumerate(word_ids):
        if wid != current_word:
            if current_embs:
                embeddings.append(torch.stack(current_embs).mean(dim=0))
            current_word = wid
            current_embs = []
        if wid is not None:
            current_embs.append(hidden_states[i])
    if current_embs:
        embeddings.append(torch.stack(current_embs).mean(dim=0))
    return torch.stack(embeddings)

def translate(en_text):
    translated = model.generate(**tokenizer(en_text, return_tensors="pt", padding=True).to(device))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def process(batch):
    results = {
        "translation": [],
        "alignments": [],
        "lv_ner_tags": [],
        "lv_ner": []
    }
    
    batch_input_str = [" ".join(i) for i in batch["tokens"]]
    encoded = tokenizer(batch_input_str, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**encoded)
    translated_sentences = tokenizer.batch_decode(translated, skip_special_tokens=True)
    translation_tokens = [i.split() for i in translated_sentences]
    results["translation"] = translation_tokens
    
    for tokens, ner_tags, translation, og in zip(batch["tokens"], batch["ner_tags"], translated_sentences, batch_input_str):
        # Alignment by nearest neighbor
        _, _, alignments = test_alignments(og, translation)

        # Project NER tags
        lv_ner_tags = [0] * len(translation.split())
        for i, j in sorted(alignments):
            if ner_tags[i] != 0:
                lv_ner_tags[j] = ner_tags[i]

        lv_ner = [ner_tag_arr[i] for i in lv_ner_tags]

        results["alignments"].append(alignments)
        results["lv_ner_tags"].append(lv_ner_tags)
        results["lv_ner"].append(lv_ner)
    
    return results

def process_ds(ds, filename):
    ds = ds.map(process, batched=True, batch_size=10)
    filtered_data = list(zip(ds['translation'], ds['lv_ner']))
    arr = np.array(filtered_data, dtype=object)
    np.save(filename, arr)
    return arr

train = process_ds(dataset['validation'], "conll_validation.npy")
print(train)
