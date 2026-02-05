## LM-CC

A code complexity metric explicitly grounded in the perspective of large language models.

LM-CC represents programs as semantic compositional hierarchies and quantifies semantic nonlinearity through a principled aggregation of compositional level and branching-induced divergence.

We demonstrate the effectiveness and practical utility of LM-CC through extensive experiments on diverse code tasks, showing significantly stronger alignment with LLM performance than existing metrics.

### Dependency
<!--Deployment and use of CodeMind require specific dependencies.--> <!--Please check if all dependencies listed on ```requirements.txt``` are installed in your environment.-->

To install all the dependencies, run the following command:

```
git clone
cd lm_cc
pip install .
pip install ./dependency/promptsource
```

### Project Structure
```
./
├── dataset/                                Raw dataset and code-rewritten dataset
    ├── humaneval/                          Dataset for execution reasoning task
    ├── humaneval-simplified/               Rewrited samples in humaneval datasset
    ├── humaneval-simplified-top60/         Rewrited samples where the original LM-CC metric ranks in the top 60%
    └── xcodeeval/
        ├── apr/                            Dataset for program repair task
        └── code_translation/               Dataset for code translation task
├── dependency/        
├── results/                                Our task outputs
    ├── humaneval-ier/                      Raw results of execution reasoning task
    ├── humaneval-ier-simplified/           Raw results of humaneval-simplified-top60
    └── xcodeeval/
        ├── apr/                            Raw results of program repair task
        ├── apr-simplified/                 Raw results of apr-simplified-top50
        ├── code_translation/               Raw results of code translation task
        └── code_translation-simplified/    Raw results of code_translation-simplified-top50           
└── scripts/                                Scripts and source code
    ├── lm-cc/                              Implementation of the lm-cc metric
    └── tasks/                              Code for task execution and evaluation
```   

### Metric Usage
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
Enter the directory:
```
cd scripts
```
#### correlations
##### Reproduce our results
```
bash correlations.sh
```
##### Start with task execution

For the execution reasoning task, we adopt the code from the original paper for generation and evaluation.

Before starting, please set the environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL_NAME`.

Then execute the corresponding task script:
```
bash program_repair.sh
bash code_translation.sh
bash execution_reasoning.sh
```

#### code rewriting
```
bash evaluate_rewriting.sh
```
