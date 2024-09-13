# Import necessary libraries
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gradio as gr

# Function to perform variational inference
def variational_inference(mu_0, sigma_0, y_values, sigma):
    # Convert y_values to a list if not already
    if isinstance(y_values, float) or isinstance(y_values, int):
        y_values = [y_values]
    elif isinstance(y_values, str):
        try:
            y_values = [float(y.strip()) for y in y_values.split(',')]
        except ValueError:
            return "Invalid input for observations. Please enter numbers separated by commas.", None

    # Input validation
    if sigma_0 <= 0 or sigma <= 0:
        return "Standard deviations must be positive numbers.", None

    # Prior distribution parameters
    mu_prior = mu_0
    sigma_prior = sigma_0

    # Sequentially update the posterior with each observation
    mu_posterior = mu_prior
    sigma_posterior_sq = sigma_prior**2

    for y_obs in y_values:
        # Likelihood parameters
        sigma_likelihood = sigma

        # Update posterior variance
        sigma_posterior_sq = 1 / (1 / sigma_posterior_sq + 1 / sigma_likelihood**2)

        # Update posterior mean
        mu_posterior = sigma_posterior_sq * (mu_posterior / sigma_posterior_sq + y_obs / sigma_likelihood**2)

    sigma_posterior = np.sqrt(sigma_posterior_sq)

    # Prepare x-axis for plotting
    x_min = min(mu_prior - 4 * sigma_prior, mu_posterior - 4 * sigma_posterior)
    x_max = max(mu_prior + 4 * sigma_prior, mu_posterior + 4 * sigma_posterior)
    x = np.linspace(x_min, x_max, 500)

    # Compute distributions
    prior_dist = norm.pdf(x, mu_prior, sigma_prior)
    posterior_dist = norm.pdf(x, mu_posterior, sigma_posterior)

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    plt.plot(x, prior_dist, label='Prior', color='blue', linestyle='--')
    plt.plot(x, posterior_dist, label='Posterior', color='red')
    plt.title('Variational Inference using Free Energy Minimization')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    # Annotate the plot with posterior mean and standard deviation
    plt.axvline(mu_posterior, color='red', linestyle=':', label='Posterior Mean')
    plt.fill_between(x, 0, posterior_dist, color='red', alpha=0.1)

    # Save the plot to a file
    plt.tight_layout()
    plot_filename = 'variational_inference_plot.png'
    plt.savefig(plot_filename)
    plt.close()

    # Prepare observations text
    observations_text = ', '.join([str(y) for y in y_values])

    # Return the posterior parameters and the plot
    result_text = (
        f"### Observations\n"
        f"{observations_text}\n\n"
        f"### Approximate Posterior Parameters\n"
        f"- **Posterior Mean (μₚ)**: {mu_posterior:.4f}\n"
        f"- **Posterior Standard Deviation (σₚ)**: {sigma_posterior:.4f}"
    )
    return result_text, plot_filename

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Probability Matrix Calculator using Variational Free Energy")
    gr.Markdown(
        "This tool performs variational inference to compute the approximate posterior distribution "
        "based on your inputs. Please provide the following parameters:"
    )
    
    with gr.Row():
        mu_0 = gr.Number(
            value=0.0, label="Prior Mean (μ₀)",
            info="Your initial estimate of the variable of interest before observing data."
        )
        sigma_0 = gr.Number(
            value=1.0, label="Prior Standard Deviation (σ₀)",
            info="The uncertainty in your prior estimate. Must be a positive number."
        )
    with gr.Row():
        y_values = gr.Textbox(
            value="1.0", label="Observations (y)",
            info="The observed data point(s). Enter multiple values separated by commas."
        )
        sigma = gr.Number(
            value=0.5, label="Observation Noise Standard Deviation (σ)",
            info="The uncertainty in your observations. Must be a positive number."
        )
    with gr.Row():
        calculate_btn = gr.Button("Calculate")
    with gr.Row():
        text_output = gr.Markdown()
    with gr.Row():
        plot_output = gr.Image(type="filepath", label="Distributions Plot")
    
    calculate_btn.click(
        fn=variational_inference,
        inputs=[mu_0, sigma_0, y_values, sigma],
        outputs=[text_output, plot_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
