# DeepCode AI Fix: Fixing Security Vulnerabilities with Large Language Models

DeepCode AI Fix is a state-of-the-art product for automatically fixing security vulnerabilities and coding errors in software systems. The key idea is to leverage program analysis to limit the LLM’s attention mechanism on the portions of code needed to perform the fix, drastically reducing the amount of required training data. Concretely, for both training and inference, rather than feeding the entire program to the LLM, we reduce its code to a much shorter snippet that contains the reported defect together with the necessary context – and use that instead.

## How can I use DeepCode AI Fix?

You can use DeepCode AI Fix in your favorite modern IDE! You will need to install Snyk IDE Extension and enable AI Fix suggestions in your Snyk account. Please refer to the documentation on Snyk's website.

- [Setting up Snyk Code in IDE](https://docs.snyk.io/scm-ide-and-ci-cd-integrations/snyk-ide-plugins-and-extensions)
- [DeepCode AI Fix](https://docs.snyk.io/scan-using-snyk/snyk-code/manage-code-vulnerabilities/fix-code-vulnerabilities-automatically)

DeepCode AI Fix is being actively maintained and improved! 🎉

## Paper
During the product development process, we have decided to share our findings with the research community and published a scientific [paper](https://arxiv.org/abs/2402.13291).

If you find our paper useful, please cite:
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

Note that the paper presents the findings from early-2023. Since then DeepCode AI Fix was significantly improved! For example, the paper trains and evaluates only on JavaScript, whereas the current product version supports more than 5 languages including Java, Python, C/C++, Go, Apex and more. To use the latest capabilities of our product/model, please refer to the previous section on using DeepCode AI Fix in your IDE.


## Setup

You need Python3 installed on your system. After that, create a virtual environment and install the dependencies:
```
cd deepcode_ai_fix
python3 -m venv dc_ai_fix_venv
source dc_ai_fix_venv/bin/activate
pip install -r requirements.txt
```

If you encounter any errors while installing requirements, it is likely because some libraries require certain packages installed system-wide. Please carefully review the error messages and install the system-wide dependencies according to your system.

Also, you might have to install torch, Nvidia drivers, and Cuda according to your GPU's requirements.

## Dataset and Models

Since DeepCode AI Fix creates a competitive advantage for Snyk, it is still unclear whether we can publish the dataset and models. Since the original product improved a lot over the paper's findings, it might be possible to expose the old model and the dataset as by now it only corresponds to a small portion of the real dataset. Please stay tuned.

Dataset schemes can be found [here](https://github.com/BBerabi/deepcode_ai_fix/blob/feat/add-training-and-inference-code/autofix/ml/lib/data_schemas.py). Below, we provide detailed explanations for each schema and field.

### LabelledDataSchema

| Fields      | Descriptions                                   |
|-------------|-------------------------------------------------|
| rule        | name of the static analysis rule, e.g. Sqli               |
| message         | message returned from the static analyzer describing the found issue                         |
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
| true_fix       | a list of boolean indicating for each prediction (corresponding indices) whether it passed the static analysis checks |
| eval_status       | a list of strings explanation for each prediction (corresponding indices) summarizing static analysis evaluation. Passed or derror message in case of failure |
| exact_match       | a list of boolean indicating for each prediction (corresponding indices) whether it exactmatly matches post_reduced |

## Fine-tuning \& Obtaining Predictions

We already provide convenient scripts to run the training and infrence. Please inspect the [training script](https://github.com/BBerabi/deepcode_ai_fix/blob/feat/add-training-and-inference-code/autofix/ml/bin/train_autofix.sh) and [inference script](https://github.com/BBerabi/deepcode_ai_fix/blob/feat/add-training-and-inference-code/autofix/ml/bin/predict_autofix.sh) carefully. 

The parameters are already set to values we mostly used in the paper but depending on which experiment you want to run, you might need to adapt a few parameters. For example, if you want to train Mixtral8x7B, you might need to enable parameter efficient fine-tuning (LoRA). We provide training details for each model in paper and you can see the available arguments in the code file [args.py](https://github.com/BBerabi/deepcode_ai_fix/blob/feat/add-training-and-inference-code/autofix/ml/lib/args.py).

### One example for training

```
env MODEL_NAME="bigcode/starcoder" NUM_EPOCHS=60 INPUT_MAX_NUM_TOKENS=512 ./autofix/ml/bin/train_autofix.py
```

### One example for inference

env MODEL_NAME="\<path_to_best_model_dir_from_training_script\> MAX_NUM_TOKENS=512 BATCH_SIZE=1 ./autofix/ml/bin/predict_autofix.sh


## Running the experiments against third-party LLMs.

In our paper, we also evaluate our approach in a few-shot learning setting using LLMs accessible only via an API (such as GPT-4). We also provide our code to run these experiments. There is one caveat. We have done these experiments by using a private API endpoint. If you have such an endpoint, then you can easily pass the URL as a command line argument. If not, you will have to slightly alter the code to use the OpenAI library directly instead of `requests` library as done in the code. This should be an easy change.

All the other details, parameters, how prompts are constructed, how fix shot examples are selected etc can be found in the code. We again provide a convenient [script](https://github.com/BBerabi/deepcode_ai_fix/blob/feat/add-training-and-inference-code/autofix/ml/bin/predict_llm.sh) to run this experiment. Please follow the instructions in the script and review the used parameter combinations depending on the model in the paper.


## What we publish and what we do not publish

DeepCode AI Fix was developed at Snyk for commercial purposes. Hence, all the code developed was part of a large confidential codebase owned by Snyk. We can not publish the entire code for two reasons:

- It would leak a lot of confidential information about how Snyk Code analyzer works
- It is not possible to decouple the relevant code from the codebase fully. 

Hence, we proceeded with the following approach. We will only publish training and inference code written in Python. CodeReduction and evaluation code (requiring MergeBack, running the analysis and computing the metrics) will not be published. Please note that the dataset contains the reduced code snippets already if it is published. So, you have reduced code snippets for our datasets but you won't be able to apply code reduction on new samples unless you implement it yourself.

Running the evaluation on your experiments is still possible even if we do not publish our code. Please refer to the section `Running Evaluations`.


## Running Evaluations

The paper describes the evaluation process in detail. This section shares useful pointers on running Snyk Code on your own predictions. You have to replicate the evaluation process yourself as we do not share the code for that. The main difference is the following: The paper runs the static analyzer directly in the first party code, hence we can not share it. To evaluate your predictions, you must run the static analyzer as a third-party, meaning using its API. Snyk has a command line interface (CLI) and you can use it to run the analyzer. Once you make the anlayzer work, we advise you to create a python script to invoke CLI commands on your predictions in an automated way. If you create one, we would highly appreciate any contributions :)


## Asking for help

We are dedicated to create a great repository that makes it possible to
- run experiments smoothly
- re-produce results
- understand the written code easily so that it can be extended

As mentioned earlier, we had to extract this code out of a large codebase while removing redundancies and confidential parts. Therefore, there can be unexpected issues while running the code. If you encounter any isses, do not hesitate to ask for help and if you fix any issues, please make a pull request. We are happy to improve the codebase!
