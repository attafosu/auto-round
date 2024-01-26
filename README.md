
# AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs

AutoRound is an advanced weight-only quantization algorithm, based on SignRound. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound with the cost of more tuning time for quantization.

## Prerequisites
- Python 3.9 or higher

## Installation
Install the necessary dependencies with the following command:
```bash
pip install -r requirements.txt
```
## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
scheme = "asym"
autoround = AutoRound(model, tokenizer, bits=4, group_size=128, scheme=scheme)
autoround.quantize()


output_dir = "/PATH/TO/SAVE/COMPRESSED/MODEL/"
autoround.export(output_dir)

##inference
## pip install intel-extension-for-transformers (for now, please install from source)
from transformers import TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig
woq_config = WeightOnlyQuantConfig(use_autoround=True,scheme=scheme)
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(output_dir, quantization_config=woq_config, trust_remote_code=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)


```
### Detailed Hyperparameters
- `model`: The PyTorch model to be quantized.
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
- `bits (int)`: Number of bits for quantization (default is 4).
- `group_size (int)`: Size of the quantization group (default is 128).
- `scheme (str)`: The quantization scheme (symmetric/asymmetric) to be used (default is "asym").
- `use_quant_input (bool)`: Whether to use the output of the previous quantized block as the input for the current block (default is True).
- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).
- `iters (int)`: Number of tuning iterations (default is 200).
- `lr (float)`: The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).
- `minmax_lr (float)`: The learning rate for min-max tuning (default is None, it will be set to lr automatically).
- `n_samples (int)`: Number of samples for tuning (default is 512).
- `seqlen (int)`: Data length of the sequence for tuning (default is 2048).
- `bs (int)`: Batch size for training (default is 8).
- `amp (bool)`: Whether to use automatic mixed precision (default is True).
- `n_blocks (int)`: Packing several blocks as one for tuning together (default is 1).
- `gradient_accumulate_steps (int)`: Number of gradient accumulation steps (default is 1).
- `low_gpu_mem_usage (bool)`: Whether to save GPU memory at the cost of a little tuning time (default is True).
- `dataset_name (str)`: The default dataset name for tuning (default is "NeelNanda/pile-10k").
- `dataset_split (str)`: The split of the dataset to be used for tuning (default is "train").
- `dataloader`: The dataloader for tuning data.
- `weight_config (dict)`: Configuration for weight quantization (default is an empty dictionary), mainly for mixed bits or mixed precision.
- `device`: The device to be used for tuning (default is "cuda:0").


## Validated Models
For wikitext2/ptb-new/c4-new ppl, we follow the code of gptq and set the sequence length to 2048. For lm-eval wikitext ppl, we adopt lm-eval. The quantization configure is W4G128.

<table border="1">
  <tr>
    <th>Model</th>
    <th>Method </th>
    <th>Acc AVG.</th>
    <th>MMLU</th>
    <th>Lamb.</th>
    <th>Hella.</th>
    <th>Wino.</th>
    <th>Piqa</th>
    <th>Truth.</th>
    <th>Open.</th>
    <th>Boolq</th>
    <th>RTE</th>
    <th>ARC-e</th>
    <th>ARC-c.</th>
    <th>wikitext2 ppl
    <th>ptb_new ppl</th>
    <th>c4_new ppl</th>
    <th>lm_eval wikitext ppl</th>
   
  </tr>

  <tr>
    <td rowspan="3">Intel/neural-chat-7b-v3 </td>
    <th>FP16</th>
    <td>67.92</td> <! acc avg -->
    <td>61.13</td> <! MMLU -->
    <td>73.03</td> <! Lambada_openai -->
    <td>66.39</td> <! Hellsaswag -->
    <td>76.40</td> <! Winogrande -->
    <td>81.01</td> <! Piqa -->
    <td>47.37</td> <! Truthfulqa -->
    <td>38.8</td> <! Openbookqa -->
    <td>86.97</td> <! Boolq -->
    <td>75.81</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.51</td> <! Arc Challenge  -->
    <td>6.00</td>  <! wikitext2 ppl  -->
    <td>48.96</td> <! ptb_new ppl  -->
    <td>9.65</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours</th>
    <td>66.90</td> <! acc avg -->
    <td>60.56</td> <! MMLU -->
    <td>72.19</td> <! Lambada_openai -->
    <td>65.28</td> <! Hellsaswag -->
    <td>75.37</td> <! Winogrande -->
    <td>81.18</td> <! Piqa -->
    <td>46.76</td> <! Truthfulqa -->
    <td>36.0</td> <! Openbookqa -->
    <td>86.91</td> <! Boolq -->
    <td>73.29</td> <! RTE -->
    <td>81.73</td> <! Arc easy -->
    <td>56.66</td> <! Arc Challenge  -->
    <td>6.21</td>  <! wikitext2 ppl  -->
    <td>59.78</td> <! ptb_new ppl  -->
    <td>10.01</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours iters1K, disable use_quant_input, minmax_lr 0.002</th>
    <td>67.70</td> <! acc avg -->
    <td>60.57</td> <! MMLU -->
    <td>73.74</td> <! Lambada_openai -->
    <td>65.62</td> <! Hellsaswag -->
    <td>77.43</td> <! Winogrande -->
    <td>80.85</td> <! Piqa -->
    <td>47.61</td> <! Truthfulqa -->
    <td>36.8</td> <! Openbookqa -->
    <td>86.94</td> <! Boolq -->
    <td>75.09</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.34</td> <! Arc Challenge  -->
    <td>6.17</td>  <! wikitext2 ppl  -->
    <td>59.12</td> <! ptb_new ppl  -->
    <td>9.83</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>


  <tr>
    <td rowspan="3">mistralai/Mixtral-8x7B-v0.1 </td>
    <th>BF16</th>
   <td>67.16</td>
    <td>69.83</td>
    <td>78.44</td>
    <td>64.89</td>
    <td>76.40</td>
    <td>82.43</td>
    <td>34.15</td>
    <td>35.40</td>
    <td>84.98</td>
    <td>71.12</td>
    <td>84.22</td>
    <td>56.91</td>
    <td>3.84</td>
    <td>19.22</td>
    <td>7.41</td>
    <td>-</td>
 
  </tr>
  <tr>
    <th>Ours</th>
    <td>65.98</td>
    <td>68.90</td>
    <td>78.11</td>
    <td>64.31</td>
    <td>74.27</td>
    <td>82.10</td>
    <td>30.97</td>
    <td>34.20</td>
    <td>84.57</td>
    <td>67.87</td>
    <td>83.96</td>
    <td>56.57</td>
    <td>4.08</td>
    <td>354</td>
    <td>7.56</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Ours iters1K, disable use_quant_input 
    <td>66.78</td>
    <td>68.68</td>
    <td>78.61</td>
    <td>64.40</td>
    <td>76.56</td>
    <td>81.99</td>
    <td>32.56</td>
    <td>34.80</td>
    <td>85.96</td>
    <td>70.76</td>
    <td>83.96</td>
    <td>56.31</td>
    <td>3.99</td>
    <td>17.65</td>
    <td>7.52</td>
    <td>-</td>
 
  </tr>
  <tr>
    <td rowspan="2">microsoft/phi-2 </td>
    <th>FP16</th>
    <td>61.80</td>
    <td>56.40</td>
    <td>62.78</td>
    <td>55.83</td>
    <td>75.77</td>
    <td>78.67</td>
    <td>31.21</td>
    <td>40.40</td>
    <td>83.36</td>
    <td>62.45</td>
    <td>80.05</td>
    <td>52.90</td>
    <td>9.71</td>
    <td>18.16</td>
    <td>14.12</td>
    <td>11.05</td>

  </tr>
  <tr>
    <th>AutoRound</th>
    <td>61.67</td>
    <td>54.57</td>
    <td>61.32</td>
    <td>55.04</td>
    <td>76.48</td>
    <td>78.89</td>
    <td>29.74</td>
    <td>40.60</td>
    <td>83.24</td>
    <td>66.43</td>
    <td>79.76</td>
    <td>52.30</td>
    <td>9.98</td>
    <td>18.67</td>
    <td>14.39</td>
    <td>11.37</td>

  </tr>
</table>


We provide a comparative analysis with other methods [link](./acc_data/REAME.md) in our accuracy data section. Notably, our approach has outperformed GPTQ with a score of 30/32 and AWQ with a score of 27/32 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, W2G128.  And the tuning costs are comparable.
### Modes passed smoke test
LaMini-GPT-124M; QWEN1-8B; OPT-125M; Bloom-560m;falcon-7b;gpt-leo-125m;stablelm-base-alpha-3b;dolly-v2-3b;mpt-7b;gpt-j-6b;chatglm2-6b


## Tips
1 Consider increasing tuning steps to achieve better results, albeit with increased tuning time. 

2 Export to GPU
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

autoround = AutoRound(model, tokenizer, bits=4, group_size=128, scheme="asym")
autoround.quantize()

## export to autogptq for gpu deployment
# please install auto-gptq https://github.com/AutoGPTQ/
output_dir = "/PATH/TO/SAVE/COMPRESSED/MODEL/"
autoround.export(output_dir, target="auto_gptq", use_triton=True) ## Utilizing Triton for 2-bit and 4-bit scenarios
# then follow auto-gptq to load the model  
```

## Known Issues
Random issues in tuning Qwen models.

ChatGlm-V1 is not supported
### Examples
cd to examples folder, install lm-eval to run the evaluation
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --enable_minmax_tuning --use_quant_input
```
- **Reduced GPU Memory Usage and Adjusted Training Batch Size:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --low_gpu_mem_usage --train_bs 1 --gradient_accumulate_steps 8
```
- **Utilizing the AdamW Optimizer:**
Include the flag `--adam`. Note that AdamW is less effective than Sign gradient descent in many scenarios we tested.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --iters 400 --lr 0.0025 --minmax_lr 0.0025
```
 `--enable_minmax_tuning` is strongly recommended 

- The transformers version required varies across different types of models. Here, the transformers version used for running models during experiments is provided as a reference.

    | Model | Transformers version |
    |  :----: | :----: |
    | EleutherAI/gpt-j-6b | 4.28/4.30/4.34/4.36 |
    | huggyllama/llama-7b | 4.28/4.30/4.34/4.36 |
    | meta-llama/Llama-2-7b-hf | 4.30/4.34/4.36 |
    | facebook/opt-6.7b | 4.28/4.30/4.34/4.36 |
    | tiiuae/falcon-7b | 4.28/4.30/4.34/4.36 |
    | mosaicml/mpt-7b | 4.28/4.30/4.34/4.36 |
    | bigscience/bloom-7b1 | 4.28/4.30/4.34/4.36 |
    | baichuan-inc/Baichuan-7B | 4.28/4.30 |
    | Qwen/Qwen-7B | 4.28/4.30/4.34/4.36 |
    | THUDM/chatglm3-6b | 4.34/4.36 |
    | mistralai/Mistral-7B-v0.1 | 4.34/4.36 |
    

## Reference
If you find SignRound useful for your research, please cite our paper:
```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```


