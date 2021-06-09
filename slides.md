---
theme: unicorn
highlighter: shiki
info: |
  ## Hands-on NLP with Hugging Face
  by María Grandury

  [My personal website](https://mariagrandury.github.io)
website: 'mariagrandury.github.io'
---

<div grid="~ cols-2" class="place-items-center">

<img style="height: 200px" src="https://huggingface.co/front/assets/huggingface_logo.svg">

<div>

# Hands-on NLP with Hugging Face

## María Grandury

</div>


</div>

<a href="https://github.com/mariagrandury" target="_blank" alt="GitHub"
  class="abs-tr m-6 text-3xl icon-btn opacity-50 !border-none !hover:text-white">
  <carbon-logo-github />
</a>


<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
website: 'mariagrandury.github.io'
---

<div grid="~ cols-2 gap-4">
<div>

<h1>About me</h1>

* 💡  Machine Learning Research Engineer

* 🎯  NLP, AI Robustness & Explainability

* 🎓  Mathematics & Physics

* 👩🏻‍💻  Trusted AI **@neurocat_GmbH**

* 🚀  Founder **@NLP_en_ES 🤗**

</div>

<div>

<h1>About this talk</h1>

* Train a Language Model: Spanish

* EsBERTa (RoBERTa-like)

* 🤗 Datasets

* 🤗 Tokenizers

* 🤗 Transformers

</div>

</div>

---
handle: 'mariagrandury'
---

# Choose the Data Set

<div grid="~ cols-2 gap-4">
<div>

<br>

The [Spanish Billion Words Corpus](https://crscardellino.github.io/SBWCE/):

- unannotated Spanish corpus
- ~1.5 billion words
- it's in [🤗 Datasets](https://huggingface.co/datasets/spanish_billion_words)!

<br>

<img style="height: 200px" class="rounded" src="https://github.com/mariagrandury/hands-on-nlp-hugging-face/raw/main/spanish_billion_words.png">

</div>
<div>

```py {1-3|6-7|10-12|all}
from datasets import load_dataset

dataset = load_dataset("spanish_billion_words")


print(len(dataset))
>> 14077588


print(dataset[23])
>> {'text': 'El señor John Dashwood no tenía la \
profundidad de sentimientos del resto de la familia...}
```

</div>
</div>

---
handle: 'mariagrandury'
---

# Tokenize the Text

<div grid="~ cols-2 gap-4">
<div>

Tokenizing:

* Split a text into (sub)words
* Convert to IDs through a look-up table

<br>

[🤗 Tokenizers](https://github.com/huggingface/tokenizers)

* Byte-Pair Encoding (BPE)
* Byte-Level BPE: GPT-2, RoBERTa
* WordPiece: BERT, ELECTRA
* SentencePiece: T5, ALBERT

</div>

<div>

<v-click>
<div>
💥 Fast State-of-the-Art Tokenizers optimized for Research and Production.

* Word tokenization
* Pre-processing:
  * Truncate
  * Pad
  * Add the special tokens

</div>
</v-click>

<v-click>
<div>

* Thanks to Rust it is extremly fast!

<br>

<div grid="~ cols-2">
<div>
<img style="height: 100px; margin-left: 50px" src="https://www.rust-lang.org/static/images/rust-logo-blk.svg">
</div>
<div>
<img style="height: 100px" class="rounded" src="https://i.gifer.com/41C.gif">
</div>
</div>

</div>
</v-click>

</div>
</div>

---
website: ''
handle: ''
---

# Train the Tokenizer

<div grid="~ cols-2 gap-4">
<div>

```py {1-5|13|7-13|12-19|21-22|all}
from tokenizers import ByteLevelBPETokenizer


# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Create iterator
def batch_iterator(batch_size=1000):
  for i in range(0, len(dataset), batch_size):
    yield dataset[i : i + batch_size]["text"]

# Customize training
tokenizer.train_from_iterator(
  iterator=batch_iterator,
  vocab_size=52_000,
  special_tokens=[
    "<s>", "</s>", "<pad>", "<unk>", "<mask>"
  ]
)

# Save files to disk
tokenizer.save_model("EsBERTa")
```

</div>

<v-click>
<div>
<div>

```py {all}
tokenizer.encode(
  "Buenos dias, me llamo Maria."
).tokens

> ['<s>',
  'Buenos', 'Ġdias', ',',
  'Ġme', 'Ġllamo',
  'ĠMaria', '.',
  '</s>']


tokenizer.encode(
  "estrambotico, despampanante, genialidad"
).tokens

> ['<s>',
  'estr', 'amb', 'ot', 'ico',
  ',',
  'Ġdesp', 'am', 'pan', 'ante',
  ',',
  'Ġgen', 'ialidad',
  '</s>']
``` 
</div>
</div>
</v-click>

</div>

---
handle: 'mariagrandury'
---

<div grid="~ cols-2">
<div>

# Transformers

## 2 Key Innovations:
* Positional Encoding
* Multi-Head Attention

## Advantages:
* All-to-all comparisons: parallelization!
* Better performance & speed
* ReLU > sigmoid, tanh
* Transfer learning: re-usable models
* [🤗 Transformers](https://github.com/huggingface/transformers) (47k ⭐)

</div>

<img style="height: 400px" border="rounded" src="https://miro.medium.com/max/700/1*BHzGVskWGS_3jEcYYi6miQ.png">
</div>

---
handle: 'mariagrandury'
---

# Configure and initialize the model


<div grid="~ cols-2 gap-4">
<div>

## RoBERTa

<br>

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

<br>

Based on Google’s BERT model (2018)
- modified key hyperparameters
- removed the next-sentence pretraining objective
- much larger mini-batches and learning rates

</div>


<div>

```py {4|1,5-12|2,14-15|3,17-18|all}
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

# Configure the model
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Create the tokenizer (Byte-Level BPE)
tokenizer = RobertaTokenizerFast.from_pretrained(
  "./EsBERTa", max_len=512
)

# Initialize the model from the config
model = RobertaForMaskedLM(config=config)
```

</div>

</div>

---
handle: 'mariagrandury'
---

# Train the Language Model

<div grid="~ cols-2 gap-4">
<div>

```py {all}
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def encode(examples):
  return tokenizer(
    examples['text'],
    truncation=True,
    padding='max_length'
  )

dataset = dataset.map(encode, batched=True)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```
</div>

<div>

```py {all}
training_args = TrainingArguments(
    output_dir="./EsBERTa",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    train_dataset=dataset,
    data_collator=data_collator,
    args=training_args,
    model=model,
)

trainer.train()
trainer.save_model("./EsBERTa")
```
</div>

</div>

---
layout: center
website: 'mariagrandury.github.io'
handle: 'mariagrandury'
---

# Thank you!

<img style="height: 200px" src="https://huggingface.co/front/assets/huggingface_logo.svg">


---