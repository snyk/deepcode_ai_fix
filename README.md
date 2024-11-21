# DeepCode AI Fix: Fixing Security Vulnerabilities with Large Language Models

DeepCode AI Fix is a state-of-the-art product for automatically fixing security vulnerabilities and coding errors in software systems. The key idea is to leverage program analysis to limit the LLM’s attention mechanism on the portions of code needed to perform the fix, drastically reducing the amount of required training data. Concretely, for both training and inference, rather than feeding the entire program to the LLM, we reduce its code to a much shorter snippet that contains the reported defect together with the necessary context – and use that instead.

## DeepCode AI Fix has gone GA! 🎉🎉🎉

After staying one and a half years in Early Access and Beta, DeepCode AI Fix has gone GA on the 29th of October 2024. See the [announcement](https://snyk.io/blog/find-auto-fix-prioritize-intelligently-snyks-ai-powered-code/) for more information and see the next section to set up and start using DeepCode AI Fix in your IDE!

## How can I use DeepCode AI Fix?

You can use DeepCode AI Fix in your favorite modern IDE! You will need to install Snyk IDE Extension and enable AI Fix suggestions in your Snyk account. Please refer to the documentation on Snyk's website.

- [Setting up Snyk Code in IDE](https://docs.snyk.io/scm-ide-and-ci-cd-integrations/snyk-ide-plugins-and-extensions)
- [DeepCode AI Fix](https://docs.snyk.io/scan-using-snyk/snyk-code/manage-code-vulnerabilities/fix-code-vulnerabilities-automatically)

DeepCode AI Fix is being actively maintained and improved! 🎉

## Paper
During the product development process, we decided to share our findings with the research community and published a scientific [paper](https://arxiv.org/abs/2402.13291).

If you find our paper useful, please cite it as follows:
```
@misc{berabi2024deepcodeaifixfixing,
      title={DeepCode AI Fix: Fixing Security Vulnerabilities with Large Language Models}, 
      author={Berkay Berabi and Alexey Gronskiy and Veselin Raychev and Gishor Sivanrupan and Victor Chibotaru and Martin Vechev},
      year={2024},
      eprint={2402.13291},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2402.13291}, 
}
```

Please note that the paper presents findings from early 2023. Since then, DeepCode AI Fix has undergone significant improvements! For example, while the paper focuses on training and evaluation using JavaScript, the current product version supports over five languages, including Java, Python, C/C++, Go, and Apex. To utilize the latest capabilities of our product, please refer to the previous section on using DeepCode AI Fix in your IDE.

## Setup

To get started, ensure that Python 3 is installed on your system. Then, follow these steps to create a virtual environment and install the necessary dependencies:

```
cd deepcode_ai_fix
python3 -m venv dc_ai_fix_venv
source dc_ai_fix_venv/bin/activate
pip install -r requirements.txt
```

If you encounter any errors during the installation of the requirements, it may be due to certain libraries needing system-wide packages. Please review the error messages carefully and install the required system-wide dependencies according to your system's specifications. For instance, you might need to install mpi packages.

```
sudo apt install libmpich-dev libopenmpi-dev

```

Additionally, you may need to install PyTorch, Nvidia drivers, and CUDA based on your GPU's requirements.

## Dataset and Models

Since DeepCode AI Fix provides a competitive advantage for Snyk, it is still undecided whether we can publicly release the dataset and models. Given the significant improvements to the original product since its initial publication, we may consider releasing the older model and dataset, as they now represent only a small portion of the current, much larger dataset. Please stay tuned for updates.

Dataset schemas are defined in the file `autofix/ml/lib/data_schemas.py`. Below, we provide detailed explanations for each schema and field.

### LabelledDataSchema (used for fine-tuning)

| Fields      | Descriptions                                   |
|-------------|-------------------------------------------------|
| rule        | name of the static analysis rule, e.g. Sqli               |
| message         | message returned from the static analyzer describing the reported issue                         |
| line_number       | line number where the static analysis report starts               |
| line_end       | line number where the static analysis report ends               |
| col_begin       | column number where the static analysis report starts               |
| col_end       | column number where the static analysis report ends               |
| pre_file       | The content of the entire file in pre-version (before fix) |
| post_file       | The content of the entire file in post-version (after fix) |
| repo       | <org_id>/<repo_name> in GitHub |
| pre_filename       | file name in pre-version (before fix) |
| post_filename       | file name in post-version (after fix) |
| pre_sha       | commit sha of pre-version |
| post_sha       | commit sha of post-version |
| pre_reduced       | Reduced code snippet containing static analysis report |
| post_reduced       | Reduced fixed code snippet |
| reduction_line_map       | Line mapping between the pre_reduced and pre_file |
| reduction_match_line_num       | Line number of the static analysis report in pre_reduced|
| language       | programming language|


Legacy fields: `jj`, `severity`, `event_id` are ignored, they were used internally for development and debugging purposes.

### PredictionSchema

Everything under `LabelledDataSchema` and

| Fields      | Descriptions                                   |
|-------------|-------------------------------------------------|
| predictions       | a list of strings containing predictions |

### EvaluationSchema

Everything under `PredictionSchema` and

| Fields      | Descriptions                                   |
|-------------|-------------------------------------------------|
| true_fix       | a list of booleans indicating for each prediction (corresponding indices) whether it passed the static analysis checks |
| eval_status       | a list of strings for each prediction (corresponding indices) summarizing static analysis evaluation. Passed or error message in case of failure |
| exact_match       | a list of booleans indicating for each prediction (corresponding indices) whether it exactly matches the target fix (post_reduced or post_file depending on the experiment) |

## Fine-tuning \& Obtaining Predictions

We provide convenient scripts for both training and inference. Please review the training script `autofix/ml/bin/train_autofix.sh` and inference script `autofix/ml/bin/predict_autofix.sh` carefully. 

The parameters are preset to the values primarily used in our paper. However, depending on the experiment you wish to conduct, you may need to adjust or add a few parameters. For example, if you're training the Mixtral8x7B model, you might need to enable parameter-efficient fine-tuning (LoRA). Detailed training information for each model is available in the paper, and you can view the available arguments in the `autofix/ml/lib/args.py` file.

To access some models, you have to accept the license aggrement on the HuggingFace UI. If the model loading crashes due to this reason, there will be useful error messages. Please follow the instructions, create an account on HuggingFace and export your HF token.

```
export HUGGING_FACE_HUB_TOKEN="<your_token>"
```

Please note that the example commands below use relative paths for reading the datasets from the `data` directory inside repo. This may or may not work depending on your setup. Please double check the arguments in the shell scripts and adjust where needed.

### One example for training

```
env MODEL_NAME="bigcode/starcoderbase-3b" NUM_EPOCHS=60 INPUT_MAX_NUM_TOKENS=512 ./autofix/ml/bin/train_autofix.sh
```

### One example for inference

```
env MODEL_NAME="path_to_best_model_dir_from_training_script" MAX_NUM_TOKENS=512 BATCH_SIZE=1 ./autofix/ml/bin/predict_autofix.sh
```

## Running the experiments against third-party LLMs

In our paper, we also evaluate our approach in a few-shot learning setting using LLMs that are accessible only via API (such as GPT-4). We provide the necessary code to run these experiments. However, there is one caveat: we conducted these experiments using a private API endpoint. If you have access to such an endpoint, you can easily pass the URL as a command-line argument. If not, you'll need to make a slight modification to the code to use the OpenAI library directly instead of the requests library, as implemented in the `autofix/ml/bin/predict_llm.py` file. This change should be straightforward.

All other details, including parameters, prompt construction, and the selection of few-shot examples, can be found in the code. We also provide a convenient script `autofix/ml/bin/predict_llm.sh` to run this experiment. Please follow the instructions in the script and review the parameter combinations used in the paper, depending on the model.

## What we publish and what we do not publish

DeepCode AI Fix was developed at Snyk for commercial purposes, and as such, the code is part of a large, confidential codebase owned by Snyk. We cannot publish the entire codebase, as it would expose a significant amount of confidential information about Snyk's technology.

To address this, we have taken the following approach: We are only publishing the training and inference code written in Python. However, the CodeReduction and evaluation code (which involves tasks like MergeBack, running analyses, and computing metrics) will not be published. If the dataset is made available, it will contain the already reduced code snippets. This means you will have access to the reduced code snippets in our datasets, but you will need to implement your own code reduction if you want to apply it to new samples.

You can still run evaluations on your experiments even without access to our unpublished code. For more details, please refer to the next section on `Running Evaluations`.

## Running Evaluations

The paper provides a detailed description of the evaluation process. This section offers guidance on running Snyk Code on your own predictions. Since we do not share the evaluation code, you will need to replicate the evaluation process yourself.

The key difference is that our evaluation runs the static analyzer directly on first-party code, which we cannot share. To evaluate your predictions, you should use the static analyzer as a third party by utilizing its API. Snyk offers a command-line interface (CLI) that you can use to run the analyzer. Please follow the steps described in the [official documentation](https://docs.snyk.io/snyk-cli/scan-and-maintain-projects-using-the-cli/snyk-cli-for-snyk-code) of Snyk.

Once you have the analyzer set up, we recommend creating a Python script to automate invoking CLI commands on your predictions. If you develop such a script, we would greatly appreciate any contributions you can make!

## Troubleshooting

We are committed to creating a repository that enables you to:
- Run experiments smoothly
- Reproduce results accurately
- Understand and extend the code easily

As noted earlier, we had to extract this code from a large codebase, removing redundancies and confidential parts. As a result, you might encounter unexpected issues while running the code. If you run into any problems, please don’t hesitate to ask for help. Additionally, if you resolve any issues, we encourage you to submit a pull request. We are eager to improve the codebase with your contributions!
