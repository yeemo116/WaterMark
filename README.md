# DS Final Project: Watermarking Framework

This project provides a basic framework for experimenting with watermark insertion and detection in deep learning models.  
Please follow the instructions below to set up your environment and begin working.

---

## Installation Guide

You can set up the environment using Anaconda.  
If you are new to Conda, please refer to the official guide:  
[Getting Started with Anaconda](https://www.anaconda.com/docs/getting-started/getting-started)

---

### Step 1: Create the Environment

Use the following command to create the environment from the provided file:

```bash
conda env create -f env.yaml
```
Use the following command to activate the environment:
```bash
conda activate DS_final_project
```

### Step 2: Run
you can run the watermark insertion by using the command.
```bash
python main.py
```

## Reference

This project is inspired by the following paper:

> **Tianhao Wang** and **Florian Kerschbaum**,  
> *"RIGA: Covert and Robust White-box Watermarking of Deep Neural Networks"*,  
> In *Proceedings of The Web Conference (WWW)*, 2021, pp. 993–1004.  
> [https://doi.org/10.1145/3442381.3450013](https://doi.org/10.1145/3442381.3450013)

This implementation is an **independent re-implementation** based on the above concept,  
and is provided **for educational and research purposes only**.
