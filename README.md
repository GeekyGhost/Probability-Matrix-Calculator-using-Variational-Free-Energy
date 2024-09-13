Don't mind the name, this is a semi repurposed project of mine based on something I was working on. 


<img width="1040" alt="Screenshot 2024-09-12 211142" src="https://github.com/user-attachments/assets/32e63878-c7b7-44b2-88e3-9cc65ba5782a">

<img width="1022" alt="Screenshot 2024-09-12 211100" src="https://github.com/user-attachments/assets/3efab684-e206-4200-81d6-81d971d62ecd">

Example

# Explanation of Variational Inference Calculator Output

The output of this variational inference calculator consists of two main parts: a textual result and a graphical plot. Let's break down each component:

## Textual Output

The text output provides the following information:

1. **Observations**: A list of the data points (y values) that were input into the model.

2. **Approximate Posterior Parameters**:
   - **Posterior Mean (μₚ)**: This is the updated estimate of the variable of interest after considering the prior knowledge and the observed data. It represents the most likely value of the parameter given the prior and the data.
   - **Posterior Standard Deviation (σₚ)**: This represents the uncertainty in the posterior estimate. A smaller value indicates higher confidence in the estimate.

## Graphical Output

The graph visualizes the prior and posterior distributions:

1. **Blue Dashed Line**: This represents the prior distribution, which is the initial belief about the parameter before observing any data.

2. **Red Solid Line**: This shows the posterior distribution, which is the updated belief about the parameter after incorporating the observed data.

3. **Red Vertical Dotted Line**: This line indicates the posterior mean (μₚ).

4. **Red Shaded Area**: This area under the posterior distribution curve represents the uncertainty in the estimate. The width of this area corresponds to the posterior standard deviation.

5. **X-axis**: Represents the possible values of the parameter of interest.

6. **Y-axis**: Represents the probability density.

## Interpretation

- The shift from the prior (blue) to the posterior (red) distribution shows how the observed data has updated our beliefs about the parameter.
- A narrower posterior distribution compared to the prior indicates that the data has reduced uncertainty about the parameter.
- The peak of the posterior distribution (where the red vertical line is) shows the most likely value of the parameter given the prior and the observed data.

This visual representation allows users to quickly grasp how their initial beliefs (prior) have been updated by the observed data to form a new understanding (posterior) of the parameter of interest."



# **Probability Matrix Calculator using Variational Free Energy*

## **Table of Contents**

- [Introduction](#introduction)
- [Mathematical Background](#mathematical-background)
- [Purpose of the Tool](#purpose-of-the-tool)
- [Installation and Setup](#installation-and-setup)
- [How to Use the Calculator](#how-to-use-the-calculator)
  - [Input Parameters](#input-parameters)
  - [Understanding the Variables](#understanding-the-variables)
  - [Performing Calculations](#performing-calculations)
- [Examples](#examples)
  - [Example 1: Single Observation](#example-1-single-observation)
  - [Example 2: Multiple Observations](#example-2-multiple-observations)
- [Interpreting the Results](#interpreting-the-results)
- [Potential Applications](#potential-applications)
- [Extending the Tool](#extending-the-tool)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**

Welcome to the **Probability Matrix Calculator using Variational Free Energy**! This tool is designed to perform **variational inference** to compute the approximate posterior distribution based on your inputs. It provides a visual and numerical representation of how your prior beliefs are updated in light of new observations.

This calculator can be particularly useful for students, researchers, and practitioners in fields such as statistics, machine learning, neuroscience, and any area where Bayesian inference is applicable.

---

## **Mathematical Background**

**Variational Inference** is a technique in Bayesian statistics that approximates probability densities through optimization. Instead of calculating the posterior distribution directly (which can be intractable for complex models), variational inference posits a simpler distribution and adjusts it to be as close as possible to the true posterior.

The **Free Energy Principle** is a concept from statistical physics and neuroscience, which in this context refers to the variational free energy that needs to be minimized to find the best approximation to the posterior distribution.

The calculator implements the following key equations for Gaussian distributions:

1. **Posterior Variance**:

   \[
   \sigma_{\text{posterior}}^2 = \left( \frac{1}{\sigma_{\text{prior}}^2} + \frac{n}{\sigma_{\text{likelihood}}^2} \right)^{-1}
   \]

   Where \( n \) is the number of observations.

2. **Posterior Mean**:

   \[
   \mu_{\text{posterior}} = \sigma_{\text{posterior}}^2 \left( \frac{\mu_{\text{prior}}}{\sigma_{\text{prior}}^2} + \frac{\sum_{i=1}^{n} y_i}{\sigma_{\text{likelihood}}^2} \right)
   \]

---

## **Purpose of the Tool**

- **Educational**: Understand how prior beliefs are updated with new data using Bayesian methods.
- **Practical**: Compute posterior distributions for problems where the prior and likelihood are Gaussian.
- **Visualization**: See the impact of observations on the posterior distribution through interactive plots.
- **Exploration**: Experiment with different parameters to gain intuition about variational inference and Bayesian updating.

---

## **Installation and Setup**

### **Prerequisites**

- **Python 3.x** installed on your system.
- Required Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `gradio`

### **Installing Dependencies**

You can install the required packages using `pip`. Open your command prompt or terminal and run:

```bash
pip install numpy scipy matplotlib gradio
```

### **Running the Application**

1. **Download the Script**

   Save the provided Python script as `variational_inference_calculator.py`.

2. **Run the Script**

   In your command prompt or terminal, navigate to the directory containing the script and run:

   ```bash
   python variational_inference_calculator.py
   ```

3. **Launch the Gradio Interface**

   After running the script, Gradio will launch a local web interface, which you can access via the URL provided in the terminal (usually `http://127.0.0.1:7860/`).

---

## **How to Use the Calculator**

### **Input Parameters**

The calculator requires the following inputs:

1. **Prior Mean (μ₀)**: Your initial estimate of the variable of interest before observing data.

2. **Prior Standard Deviation (σ₀)**: Represents the uncertainty in your prior estimate. Must be a positive number.

3. **Observations (y)**: The observed data point(s). You can enter a single value or multiple values separated by commas (e.g., `1.0, 2.5, 3.2`).

4. **Observation Noise Standard Deviation (σ)**: The uncertainty associated with your observations. Must be a positive number.

### **Understanding the Variables**

#### **1. Prior Mean (μ₀)**

- **Definition**: The expected value of the variable based on prior knowledge.
- **How to Determine**: Use historical data, expert opinion, or a neutral default value (like zero) if unsure.
- **Impact**: Sets the center of your prior distribution.

#### **2. Prior Standard Deviation (σ₀)**

- **Definition**: Measures the spread or uncertainty of your prior belief.
- **How to Determine**:
  - **High σ₀**: Indicates high uncertainty about the prior mean.
  - **Low σ₀**: Indicates strong confidence in the prior mean.
- **Impact**: Affects how much weight the prior has compared to the observations.

#### **3. Observations (y)**

- **Definition**: The data points you've collected.
- **How to Determine**: Based on measurements, experiments, or data collection.
- **Format**: Enter numbers separated by commas (e.g., `1.2, 2.3, 0.9`).

#### **4. Observation Noise Standard Deviation (σ)**

- **Definition**: Represents the measurement error or variability in your observations.
- **How to Determine**:
  - Based on instrument precision or estimated variability.
  - Smaller σ indicates more precise measurements.
- **Impact**: Influences how much each observation adjusts the posterior.

### **Performing Calculations**

1. **Enter the Input Parameters**

   - Fill in the input fields with your values.
   - Ensure standard deviations (`σ₀` and `σ`) are positive numbers.

2. **Click "Calculate"**

   - The calculator will process your inputs.
   - It performs variational inference to compute the approximate posterior distribution.

3. **View the Results**

   - **Approximate Posterior Parameters**: Displays the posterior mean and standard deviation.
   - **Distributions Plot**: Shows the prior and posterior distributions on a graph.

---

## **Examples**

### **Example 1: Single Observation**

**Scenario**: You have a prior belief that the average height of a plant species is 50 cm with some uncertainty. You measure one plant and find it to be 55 cm tall.

**Input Parameters**:

- **Prior Mean (μ₀)**: `50`
- **Prior Standard Deviation (σ₀)**: `5`
- **Observations (y)**: `55`
- **Observation Noise Standard Deviation (σ)**: `2`

**Steps**:

1. Enter the values into the calculator.
2. Click "Calculate".
3. Review the updated posterior mean and standard deviation.

**Expected Outcome**:

- The posterior mean will shift closer to 55 cm but won't equal it due to the prior influence.
- The posterior standard deviation will be smaller than the prior, indicating reduced uncertainty.

### **Example 2: Multiple Observations**

**Scenario**: You have no strong prior belief about a variable. You collect multiple measurements: 1.2, 0.9, 1.1.

**Input Parameters**:

- **Prior Mean (μ₀)**: `0` (neutral prior)
- **Prior Standard Deviation (σ₀)**: `10` (high uncertainty)
- **Observations (y)**: `1.2, 0.9, 1.1`
- **Observation Noise Standard Deviation (σ)**: `0.1`

**Steps**:

1. Enter the values into the calculator.
2. Click "Calculate".
3. Review the updated posterior mean and standard deviation.

**Expected Outcome**:

- The posterior mean will be close to the average of the observations.
- The posterior standard deviation will be significantly smaller, reflecting increased confidence.

---

## **Interpreting the Results**

- **Posterior Mean (μₚ)**: Represents the updated estimate of the variable after considering the observations.
- **Posterior Standard Deviation (σₚ)**: Indicates the uncertainty in the updated estimate.
  - A smaller σₚ means higher confidence.
- **Distributions Plot**:
  - **Prior Distribution**: Shown as a dashed blue line.
  - **Posterior Distribution**: Shown as a solid red line.
  - The plot helps visualize how the observations have shifted and shaped your belief.

---

## **Potential Applications**

- **Statistical Modeling**: Updating model parameters with new data.
- **Machine Learning**: Parameter estimation in Bayesian models.
- **Neuroscience**: Understanding perception as Bayesian inference.
- **Economics and Finance**: Adjusting forecasts with new market data.
- **Engineering**: Sensor fusion and uncertainty quantification.

---

## **Extending the Tool**

- **Different Distributions**: Modify the code to handle non-Gaussian priors and likelihoods.
- **Multivariate Analysis**: Extend to multivariate variables and observations.
- **Real-time Updating**: Implement streaming data handling for sequential updates.
- **Advanced Visualizations**: Add interactive plots with zoom and hover functionalities.

---

## **Troubleshooting**

### **Common Issues**

1. **Invalid Input for Observations**

   - **Problem**: Error message stating invalid input.
   - **Solution**: Ensure observations are numbers, separated by commas, without any letters or special characters.

2. **Negative Standard Deviations**

   - **Problem**: Standard deviations must be positive numbers.
   - **Solution**: Check that `σ₀` and `σ` are greater than zero.

3. **No Output Displayed**

   - **Problem**: After clicking "Calculate", nothing happens.
   - **Solution**:
     - Check the console for error messages.
     - Ensure all required packages are installed.
     - Restart the application.

4. **Plot Not Showing Correctly**

   - **Problem**: The distributions plot is blank or incomplete.
   - **Solution**:
     - Verify that the input values make sense (e.g., standard deviations are not too large).
     - Adjust the prior and observation values to ensure the distributions overlap within the plot range.

---

## **Additional Resources**

- **Books**:
  - *Pattern Recognition and Machine Learning* by Christopher M. Bishop
  - *Bayesian Data Analysis* by Andrew Gelman et al.
- **Online Courses**:
  - Coursera: *Probabilistic Graphical Models* by Stanford University
  - edX: *Bayesian Statistics* by University of California, Santa Cruz
- **Research Papers**:
  - Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- **Websites**:
  - [Wikipedia: Variational Bayesian Methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
  - [Gradio Documentation](https://gradio.app/)

---

## **Contributing**

Contributions to improve this tool are welcome! Here are some ways you can contribute:

- **Report Bugs**: If you find any issues or bugs, please report them.
- **Suggest Features**: Have an idea to make the tool better? Let us know!
- **Code Contributions**: Fork the repository, make changes, and submit a pull request.

---

## **License**

This project is licensed under the MIT License. You are free to use, modify, and distribute this software.

---

**Thank you for using the Probability Matrix Calculator! We hope it enhances your understanding and application of variational inference and Bayesian methods.**

---

# **Appendix: Code Explanation**

Below is a brief explanation of the key components of the code for those interested in understanding how the calculator works.

### **Imports**

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gradio as gr
```

- **numpy**: For numerical computations.
- **scipy.stats**: Provides probability distributions.
- **matplotlib.pyplot**: For plotting the distributions.
- **gradio**: For building the interactive user interface.

### **Variational Inference Function**

```python
def variational_inference(mu_0, sigma_0, y_values, sigma):
    # Input parsing and validation
    # Prior and likelihood parameters
    # Sequential updating of posterior
    # Calculation of posterior mean and variance
    # Plotting the distributions
    # Preparing the output
    return result_text, plot_filename
```

- **Input Parsing**: Converts observations into a list of numbers.
- **Input Validation**: Ensures standard deviations are positive.
- **Sequential Updating**: Updates the posterior for each observation.
- **Plotting**: Visualizes the prior and posterior distributions.

### **User Interface with Gradio**

```python
with gr.Blocks() as demo:
    # Markdown descriptions
    # Input fields with explanations
    # Calculate button
    # Output fields for results and plot
```

- **gr.Blocks**: Creates a web interface layout.
- **Input Fields**: Collects user inputs with helpful tooltips.
- **Output Fields**: Displays the results and plot.

### **Launching the Application**

```python
if __name__ == "__main__":
    demo.launch()
```

- **demo.launch()**: Starts the Gradio interface, making the calculator accessible via a web browser.

---

# **Frequently Asked Questions (FAQ)**

**Q1: Can I use this tool for non-Gaussian distributions?**

A: Currently, the tool is designed for Gaussian priors and likelihoods. Extending it to non-Gaussian distributions would require modifying the variational inference calculations accordingly.

**Q2: How does the observation noise standard deviation affect the results?**

A: A smaller observation noise standard deviation means you trust your observations more, so they have a greater impact on the posterior. A larger standard deviation means the observations are less reliable, and the posterior relies more on the prior.

**Q3: What happens if I have conflicting observations?**

A: The calculator updates the posterior sequentially. Conflicting observations will be balanced based on their values and the observation noise standard deviation. The posterior standard deviation may increase if there's significant disagreement.

**Q4: Can I input a large number of observations?**

A: Yes, but be mindful that very large datasets may slow down the computation and plotting. For large datasets, consider summarizing the data or modifying the code to handle batch updates more efficiently.

**Q5: Is the calculator suitable for real-time data analysis?**

A: While the calculator can handle sequential updates, it's not optimized for real-time streaming data. For real-time applications, further development and optimization would be necessary.
