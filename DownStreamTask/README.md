This folder contains the datasets of all downstream tasks we used: [Word in Context (WiC)](https://pilehvar.github.io/wic/), [Winograd Schema Challenge (WSC)](https://cs.nyu.edu/˜davise/papers/WinogradSchemas/WS.html), [Sense Making (SM), Sense Making Reasoning (SMR)](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation), and [HellaSwag](https://rowanzellers.com/hellaswag/).


These datasets are all sourced from the official website, so for convenience, we put it here and you can use the functions inside the utils.py file to read the datasets.
| Task | Examples |
|:---:|:---|
| Word in Context(WiC) | **Context 1**: Room and <u>board</u>. <br> **Context 2**: He nailed <u>boards</u> across the windows. <br> **Sense match**: False |
| Winograd Schema Challenge (WSC) | **Text**: Mark told <u>Pete</u> many lies about himself, which Pete included in his book. <u>He</u> should have been more truthful.  <br>**Coreference**: False |
| Sense Making (SM) | **Sentence1**: money can be used for buying cars.  <br>**Sentence2**: money can be used for buying stars.  <br>**Label**: Sentence1 |
| Sense Making Reasoning (SMR) | **Sentence**: "he put an elephant into the fridge", -> because  <br>**OptionA**: an elephant is much bigger than a fridge .  <br>**OptionB**: elephants are usually gray...  <br>**OptionC**: an elephant cannot eat a fridge .  <br>**Label**: OptionA |
| HellaSwag | **Sentence**: A carved pumpkin with a light in it glows on a counter. Supplies for carving are then shown.  <br>**Ending1**: A woman cuts the top off the pumpkin, emptying the seeds.  <br>**Ending2**: she cuts down all the pieces and dumps them in a trash bin in the end.  <br>**Ending3**: she then carves the traced lines to cut out the design.  <br>**Ending4**: she tapes the top shut as the continue carving the pumpkin. <br>**Label**: Ending1 |