# HTTF
Repository for Hail to the Thief: Poisoning GRPO


All code is in the src/ folder

Tests can be ran by:

```
python [filename] [device index] [benign or malicious (0 or 1)]
```

Some defences require a huggignface token for the llama model

Yoiu will need transformers & pytorch to run this repo

The tests:
Hail to the thief vertical - runner_vertical.py
Hail to the thief horizontal - runner_horizontal.py
2+2=5 vertical - math_vertical.py

The defended versions include _defended