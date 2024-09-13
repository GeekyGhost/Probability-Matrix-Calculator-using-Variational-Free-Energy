# Import necessary libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gradio as gr
from io import BytesIO
import base64

# Function to perform variational inference
def variational_inference(mu_0, sigma_0, y_values, sigma):
    """
    Perform variational inference to compute the approximate posterior distribution.

    Parameters:
    - mu_0 (float): Prior mean (μ₀)
    - sigma_0 (float): Prior standard deviation (σ₀)
    - y_values (str): Observations (y), comma-separated if multiple
    - sigma (float): Observation noise standard deviation (σ)

    Returns:
    - result_text (str): Markdown-formatted text with posterior parameters
    - plot_image (str): Base64-encoded PNG image of the distributions plot
    """
    # Input validation for standard deviations
    if sigma_0 <= 0 or sigma <= 0:
        return "❌ **Error**: Standard deviations (σ₀ and σ) must be positive numbers.", None

    # Parse observations
    if isinstance(y_values, (float, int)):
        y_values = [y_values]
    elif isinstance(y_values, str):
        try:
            # Split by commas and convert to floats
            y_values = [float(y.strip()) for y in y_values.split(',') if y.strip() != '']
            if not y_values:
                return "❌ **Error**: No valid observations provided. Please enter numbers separated by commas.", None
        except ValueError:
            return "❌ **Error**: Invalid input for observations. Please enter numbers separated by commas.", None
    else:
        return "❌ **Error**: Invalid input type for observations.", None

    # Prior distribution parameters
    mu_prior = mu_0
    sigma_prior = sigma_0

    # Initialize posterior parameters
    mu_posterior = mu_prior
    sigma_posterior_sq = sigma_prior**2

    # Sequentially update the posterior with each observation
    for y_obs in y_values:
        # Update posterior variance
        sigma_posterior_sq = 1 / (1 / sigma_posterior_sq + 1 / sigma**2)
        sigma_posterior = np.sqrt(sigma_posterior_sq)

        # Update posterior mean
        mu_posterior = sigma_posterior_sq * (mu_posterior / sigma_posterior_sq + y_obs / sigma**2)

    # Prepare x-axis for plotting
    x_min = min(mu_prior - 4 * sigma_prior, mu_posterior - 4 * sigma_posterior)
    x_max = max(mu_prior + 4 * sigma_prior, mu_posterior + 4 * sigma_posterior)
    x = np.linspace(x_min, x_max, 500)

    # Compute distributions
    prior_dist = norm.pdf(x, mu_prior, sigma_prior)
    posterior_dist = norm.pdf(x, mu_posterior, sigma_posterior)

    # Plot the distributions using in-memory buffer
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_dist, label='Prior (μ₀, σ₀)', color='blue', linestyle='--')
    plt.plot(x, posterior_dist, label='Posterior (μₚ, σₚ)', color='red')
    plt.title('Variational Inference using Free Energy Minimization')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    # Annotate the plot with posterior mean and standard deviation
    plt.axvline(mu_posterior, color='red', linestyle=':', label='Posterior Mean (μₚ)')
    plt.fill_between(x, 0, posterior_dist, color='red', alpha=0.1)

    # Save the plot to an in-memory buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_image = base64.b64encode(buf.read()).decode()

    # Prepare observations text
    observations_text = ', '.join([f"{y:.4f}" for y in y_values])

    # Prepare result text with Markdown formatting
    result_text = (
        f"### 📊 **Variational Inference Results**\n\n"
        f"**Observations (y):** {observations_text}\n\n"
        f"**Approximate Posterior Parameters:**\n"
        f"- **Posterior Mean (μₚ):** {mu_posterior:.4f}\n"
        f"- **Posterior Standard Deviation (σₚ):** {sigma_posterior:.4f}\n\n"
        f"**Interpretation:**\n"
        f"- The **posterior mean** represents the updated estimate after considering the observations.\n"
        f"- The **posterior standard deviation** indicates the uncertainty in this estimate."
    )

    # Convert the image to base64 for Gradio display
    plot_markdown = f"![Distributions Plot](data:image/png;base64,{plot_image})"

    # Combine result text and plot
    full_result = result_text + "\n\n" + plot_markdown

    return full_result, None  # Gradio can render Markdown with images inline

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧮 Probability Matrix Calculator using Variational Free Energy")
    gr.Markdown(
        """
        This tool performs **variational inference** to compute the approximate posterior distribution 
        based on your inputs. It provides a visual and numerical representation of how your prior 
        beliefs are updated in light of new observations.
        """
    )
    
    with gr.Row():
        mu_0 = gr.Number(
            value=0.0, 
            label="📈 Prior Mean (μ₀)",
            info="Your initial estimate of the variable of interest before observing data."
        )
        sigma_0 = gr.Number(
            value=1.0, 
            label="📉 Prior Standard Deviation (σ₀)",
            info="The uncertainty in your prior estimate. Must be a positive number."
        )
    
    with gr.Row():
        y_values = gr.Textbox(
            value="1.0", 
            label="📝 Observations (y)",
            placeholder="Enter numbers separated by commas (e.g., 1.0, 2.5, 3.2)",
            info="The observed data point(s). Enter multiple values separated by commas."
        )
        sigma = gr.Number(
            value=0.5, 
            label="🔍 Observation Noise Std Dev (σ)",
            info="The uncertainty in your observations. Must be a positive number."
        )
    
    with gr.Row():
        calculate_btn = gr.Button("✅ Calculate", variant="primary")
    
    with gr.Row():
        text_output = gr.Markdown()
    
    with gr.Row():
        plot_output = gr.Markdown()
    
    calculate_btn.click(
        fn=variational_inference,
        inputs=[mu_0, sigma_0, y_values, sigma],
        outputs=[text_output, plot_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
