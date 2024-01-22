
# AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs

AutoRound is an advanced weight-only quantization algorithm, based on SignRound. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound. However, it comes at the cost of approximately 2.5 times the tuning runtime.

## Prerequisites
- Python 3.9 or higher


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
    
Please note that all experiments in the SignRound+ technical report were conducted using transformers version 4.34.1.



## Installation
Install the necessary dependencies with the following command:
```bash
pip install -r requirements.txt
```
## Uasage
```python
from auto_round import AutoRound
model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
autoround = round(model, tokenizer, num_bits=4, group_size=128, scheme="asym")
fake_qdq_model = autoround.quantize()

## export to gpu
# packed_folder = "./tmp_autoround_packed"
# autoround.export_to_autogptq("packed_folder")

```
### Detailed Hyperparameters
- `model`: The PyTorch model to be quantized.
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
- `bits (int)`: Number of bits for quantization (default is 4).
- `group_size (int)`: Size of the quantization group (default is 128).
- `scheme (str)`: The quantization scheme (symmetric/asymmetric) to be used (default is "asymmetric").
- `use_quant_input (bool)`: Whether to use the output of the previous quantized block as the input for the current block (default is True).
- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).
- `iters (int)`: Number of tuning iterations (default is 200).
- `lr (float)`: The learning rate for rounding value (default is 0.005).
- `minmax_lr (float)`: The learning rate for min-max tuning (default is None).
- `n_samples (int)`: Number of samples for tuning (default is 512).
- `seqlen (int)`: Data length of the sequence for tuning.
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
Include the flag `--adam`. Note that AdamW may be  less effective than Sign gradient descent in many scenarios.

- **Running the Original SignRound:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name facebook/opt-125m --amp --bits 4 --group_size -1 --iters 400 --lr 0.0025 --minmax_lr 0.0025
```
 `--enable_minmax_tuning` is strongly recommended 



## Tips
Consider increasing tuning steps and adjusting the learning rate based on a scaling law to achieve better results, albeit with increased tuning time. For instance, at step 800, a learning rate of 0.00125 could be employed.


## Known Issues
Auto Rounding may encounter random issues with Qwen models.

ChatGlm-V1 is not supported

We are working on exporting the quantized model to HF format

Cpu kernel will be supported soon

## Validated Models

[//]: # ()
[//]: # (| W4G128                      | MMLU | Lamb. | Hella. | Wino. | Piqa | Truth. | Open. | Boolq | RTE | ARC-e | ARC-c. | AVG. |)

[//]: # (|-----------------------------|------|-------|--------|-------|------|--------|-------|-------|-----|-------|--------|------|)

[//]: # (|                             |      |       |        |       |      |        |       |       |     |       |        |      |)

[//]: # (| mistralai/Mixtral-8x7B-v0.1 |      |       |        |       |      |        |       |       |     |       |        |      |)

[//]: # (|-----------------------------|------|-------|--------|-------|------|--------|-------|-------|-----|-------|--------|------|)

[//]: # (| microsoft/phi-2             |      |       |        |       |      |        |       |       |     |       |        |      |)

For a fair comparison, we utilized 512 samples from Pile-10k for all methods during calibration. Due to memory constraints, we maintained the original sequence length of 512 for AWQ, while for GPTQ and our approach,  a sequence length of 2048 is used. We have enalbed act-order and true-seqential in GPTQ, and the notation GPTQ* indicates that we adjusted the random seed or data preprocessing to address issues related to the non-positive definite Hessian matrix or other issues.
![](./figs/W4G-1.png)
![](./figs/W4G128.png)
![](./figs/W3G128.png)
![](./figs/W2G128.png)

Mistral-7b  done

LLaMAV1 done

LLaMAv2 done

PI2   done

mixstral-7Bx8 done

LaMini-GPT-124M done

QWEN1-8B done,but has random issue

OPT-125M done

Bloom-560 smoke test done

falcon-7b smoke test done

gpt-leo-125m smoke test done

stablelm-base-alpha-3b smoke test done

dolly-v2-3b smoke test done

mpt-7b smoke test done

gpt-j-6b smoke test done

chatglm2-6b smoke test done



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

