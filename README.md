# DeepCode AI Fix: Fixing Security Vulnerabilities with Large Language Models

DeepCode AI Fix is a state-of-the-art production system for automatically fixing security vulnerabilities and coding errors in programs. The key idea is to leverage program analysis to limit the LLM’s attention mechanism on the portions of code needed to perform the fix, drastically reducing the amount of required training data. Concretely, for both training and inference, rather than feeding the entire program to the LLM, we reduce its code to a much shorter snippet that contains the reported defect together with the necessary context – and use that instead.

## Live Product

DeepCode AI Fix started as a research project at Snyk but now it is a live product being used by developers. DeepCode AI Fix is still being actively developed and improved! To start using Snyk and DeepCode AI, please follow the instructions on the official Snyk website. Some usefuls links can be found below.

## Paper
During the product development process, we have decided to share our findings with the research community and published a scientific paper.

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

Note that the paper presents the findings from mid-2023. Since then DeepCode AI Fix has improved a lot! For instance, the paper only trains and evaluates on JavaScript, whereas DeepCode AI Fix currently supports more than 5 languages including Java, Python, C/C++, Go, Apex and more.


## Setup

You need Python3 installed on your system. After that, create a virtual environment and install the dependencies:
```
python3 -m venv dc_ai_fix_venv
source dc_ai_fix_venv/bin/activate
pip install -r requirements.txt
```

Note that you might have to install torch, Nvidia drivers, and Cuda according to your GPU's requirements.

## Dataset and Models

Since DeepCode AI Fix creates a competitive advantage for Snyk, it is still unclear whether we can publish the dataset and models. Since the original product improved a lot over the paper's findings, it might be possible to expose the old model and the dataset as by now it only corresponds to a small portion of the real dataset. Please stay tuned.

Dataset schemes can be found here. Below, we provide detailed explanations for each fields.

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


Legacy fields: jj, severity, event_id

| predictions       | a list of strings containing predictions |
| true_fix       | a list of boolean indicating for each prediction (corresponding indices) whether it passed the static analysis checks |
| eval_status       | a list of strings explanation for each prediction (corresponding indices) summarizing static analysis evaluation. Passed or derror message in case of failure |
| exact_match       | a list of boolean indicating for each prediction (corresponding indices) whether it exactmatly matches post_reduced |


