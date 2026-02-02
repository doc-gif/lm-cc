## LM-CC

A code complexity metric explicitly grounded in the perspective of large language models.

### Dependency

<!--Deployment and use of CodeMind require specific dependencies.--> <!--Please check if all dependencies listed on ```requirements.txt``` are installed in your environment.-->

To install all the dependencies, run the following command:

```
git clone
cd lm_cc
pip install .
```

### Usage
Default: Download the model from Hugging Face and load it in full precision.

``` python
from lm_cc import get_code_lmcc
cc = get_code_lmcc(code)
print("LM-CC:", cc)
```

For custom parameters (e.g., loading the model in half precision), follow the steps below:

``` python
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


### Reproduce Results

```
cd scripts
```

#### correlations

Use our token entropy file to calculate metrics and correlation coefficients:

```
python correlation.py --output-dir ouroutput --task [program_repair/code_translation/execution_reasoning] 
```

<!-- Or start with calculating token entropy:

```
python correlation.py --task [program_repair/code_translation/execution_reasoning]
``` -->

Cached metric files can be used to speed up repeated runs:
```
python correlation.py --task program_repair --cache
```


#### code rewriting