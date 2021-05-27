---
theme: unicorn
highlighter: shiki
info: |
  ## Hands-on NLP with Hugging Face
  by María Grandury

  [My personal website](https://mariagrandury.github.io)
website: 'mariagrandury.github.io'
handle: 'mariagrandury'
---

<div grid="~ cols-2" class="place-items-center">

<img style="height: 250px" src="https://huggingface.co/front/assets/huggingface_logo.svg">

<div>

# Hands-on NLP with Hugging Face

## WomenTech Global Conference

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

# Huggingface Tokenizers

<div grid="~ cols-2 gap-4">
<div>

https://github.com/huggingface/tokenizers

💥 Fast State-of-the-Art Tokenizers optimized for Research and Production.

* Word tokenization
* Pre-processing:
  * Truncate
  * Pad
  * Add the special tokens

<v-click>
Thanks to Rust its extremly fast!

<img src="https://www.rust-lang.org/static/images/rust-logo-blk.svg">
</v-click>

</div>
<v-click>
<div>

<img class="rounded" src="https://i.gifer.com/41C.gif">

</div>
</v-click>
</div>

---

# Train a Tokenizer

```py {1-5|7-8|10-18|20-21|all}
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("EsperBERTo")
``` 

---

<div grid="~ cols-2">
<div>

# Transformers

https://github.com/huggingface/transformers

## Advantages w.r.t. RNNs:
* more easily parallelized
* better performance & speed
* capture much longer dependencies
* transfer learning
* coolerr name

## Architecture:
* Positional Encoding
* Masking
* Multi-Head Attention
</div>

<img border="rounded" src="https://miro.medium.com/max/700/1*BHzGVskWGS_3jEcYYi6miQ.png">
</div>

---

# Configure and initialize the model

```py {1,5-12|2,14-15|3,17-18|all}
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast

# Configure the model
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Create the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

# Initialize the model from the config
model = RobertaForMaskedLM(config=config)
```

---

# Build the Training Dataset

```py {1, 5-6|2,8-13|3,15-18|all}
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling


tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```

---

# Train a Language Model

```py {1|3-11|13-18|20|21|all}
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./EsperBERTo")
```

---

# Fine-tune the LM on a downstream task

```py {1|4-10|12-18|all}
from transformers import TokenClassificationPipeline, pipeline


MODEL_PATH = "./models/EsperBERTo-small-pos/"

nlp = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
)

nlp("Mi estas viro kej estas tago varma.")

# {'entity': 'PRON', 'score': 0.9979867339134216, 'word': ' Mi'}
# {'entity': 'VERB', 'score': 0.9683094620704651, 'word': ' estas'}
# {'entity': 'VERB', 'score': 0.9797462821006775, 'word': ' estas'}
# {'entity': 'NOUN', 'score': 0.8509314060211182, 'word': ' tago'}
# {'entity': 'ADJ', 'score': 0.9996201395988464, 'word': ' varma'}
```

---
layout: center
logoHeader: 'https://huggingface.co/front/assets/huggingface_logo.svg'
website: 'mariagrandury.github.io'
handle: 'mariagrandury'
---

# Thank you

link to hugging face repo

---
