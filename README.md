## LM-CC

A code complexity metric explicitly grounded in the perspective of large language models.

### Dependency

<!--Deployment and use of CodeMind require specific dependencies.--> <!--Please check if all dependencies listed on ```requirements.txt``` are installed in your environment.-->

To install all the dependencies, run the following command:

```
git clone
cd lm_cc
pip install .
pip install ./dependency/promptsource
```

### Usage
Default: Download the model from Hugging Face and load it in full precision.

``` 
from lm_cc import get_code_lmcc
cc = get_code_lmcc(code)
print("LM-CC:", cc)
```

For custom parameters (e.g., loading the model in half precision), follow the steps below:

``` 
from lm_cc import get_code_lmcc, CodeBlockProcessor, TokenEntropyCalculator
processor = CodeBlockProcessor()
calculator = TokenEntropyCalculator(
    model_name = "codellama/CodeLlama-7b-hf",
    cache_dir = "",    
    device_map="auto", 
    float_type=torch.float16,
)
cc = get_code_lmcc(code, calculator=calculator, processor=processor)
print("LM-CC:", cc)
```


### Experiment

```
cd scripts
```

#### correlations
* To reproduce our results:
```
bash correlations.sh
```
* Start with task execution:
For the execution reasoning task, we adopt the code from the original paper for generation and evaluation.
```
bash program_repair.sh
bash code_translation.sh
bash execution_reasoning.sh
```


#### code rewriting