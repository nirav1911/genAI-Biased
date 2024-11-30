# genAI-Biased

## Installation

Install required libraries:

```bash
pip install transformers datasets scipy scikit-learn numpy
pip3 install torch torchvision
```

## Winobias Testing

To test different Large Language Models (LLMs) against the WinoBias dataset, you can use a framework like LangTest, which simplifies the evaluation process.

### Step 1: Install Required Libraries

```bash
pip install "langtest[ai21,openai]==1.7.0"
```

### Step 2: Set Up Environment Variables

Set your API keys for the LLMs you want to test:

```python
import os

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"
```

### Step 3: Import LangTest and Configure the Harness

```python
from langtest import Harness

# Configure the harness for WinoBias testing
harness = Harness(
    task="wino-bias",
    model={"model": "text-davinci-003", "hub": "openai"},
    data={"data_source": "Wino-test"}
)

# Generate test cases and run evaluations
harness.generate().run().report()
```

**Explanation:**
- LangTest: A library that provides tools for testing LLMs on various tasks, including bias detection.
- Harness Configuration: Specifies the task (`wino-bias`), model details (e.g., `text-davinci-003` from OpenAI), and data source (`Wino-test`).

### Step 4: Analyze Results

After running the tests, LangTest will generate a report detailing the model's performance on pro-stereotypical and anti-stereotypical scenarios. This includes metrics like accuracy and F1 score differences between these conditions.

**Additional Considerations:**
- Testing Multiple Models: You can easily switch models by changing the `model` parameter in the `Harness` configuration.
- Advanced Metrics: Consider implementing additional bias metrics like skew and stereotype to gain deeper insights into model biases.
- Data Augmentation: Use techniques like gender swapping or data augmentation to further analyze and mitigate biases.

## Modifying LangTest Evaluation to Include More LLMs

1. Install additional LLM libraries:

```bash
pip install "langtest[ai21,openai,cohere]==1.7.0"
```

2. Set up API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"
os.environ["AI21_API_KEY"] = "<YOUR_AI21_KEY>"
os.environ["COHERE_API_KEY"] = "<YOUR_COHERE_KEY>"
```

3. Modify the Harness configuration:

```python
from langtest import Harness

models = [
    {"model": "text-davinci-003", "hub": "openai"},
    {"model": "j2-jumbo-instruct", "hub": "ai21"},
    {"model": "command-xlarge-nightly", "hub": "cohere"}
]

for model in models:
    harness = Harness(
        task="wino-bias",
        model=model,
        data={"data_source": "Wino-test"}
    )
    harness.generate().run().report()
```

4. Run the evaluation: Execute the modified script to test multiple LLMs against the WinoBias dataset.

Sources
