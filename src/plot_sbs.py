import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Define colors from LaTeX definitions (HTML hex codes)
teal = '#2FBEAD'
lightgrey = '#E8ECEE'
beige = '#E8D5C4'
lightgreen_new = '#BDF5BC' # Renamed to avoid conflict with previous lightgreen
lightpink = '#F5BCD9'

# Data from the tables
data = {
    "SynthDetoxM Subset\n (Ours) vs MultiParaDetox": {
        "Overall": {"q1": 679, "tie": 706, "q2": 290},
        "de": {"q1": 228, "tie": 232, "q2": 86},
        "ru": {"q1": 192, "tie": 277, "q2": 89},
        "es": {"q1": 259, "tie": 197, "q2": 115},
    },
    "SynthDetoxM Full\n (Ours) vs MultiParaDetox": {
        "Overall": {"q1": 767, "tie": 673, "q2": 225},
        "de": {"q1": 244, "tie": 223, "q2": 77},
        "ru": {"q1": 242, "tie": 244, "q2": 64},
        "es": {"q1": 281, "tie": 206, "q2": 84},
    },
    "SynthDetoxM Full\n (Ours) vs SynthDetoxM Subset\n (Ours)": {
        "Overall": {"q1": 516, "tie": 803, "q2": 369},
        "de": {"q1": 166, "tie": 247, "q2": 133},
        "ru": {"q1": 157, "tie": 310, "q2": 106},
        "es": {"q1": 193, "tie": 246, "q2": 130},
    },
    "SynthDetoxM Full\n (Ours) vs SynthDetoxM Full\n (Ours) \n+\n MultiParaDetox": {
        "Overall": {"q1": 644, "tie": 815, "q2": 196},
        "de": {"q1": 201, "tie": 255, "q2": 84},
        "ru": {"q1": 199, "tie": 301, "q2": 54},
        "es": {"q1": 244, "tie": 259, "q2": 58},
    },
}

languages = ["Overall", "de", "ru", "es"]
comparisons = list(data.keys())
bar_labels = ["Model A", "Tie", "Model B"]
bar_colors = [teal, lightgrey, beige] # Use the defined colors

fixed_plot_width = 1000  # Define a fixed width for all plots


for lang in languages:
    fig = plt.figure(figsize=(18, 6)) # Create a new figure for each language, adjust height as needed
    ax = fig.add_subplot(1, 1, 1) # Add a subplot to the new figure

    comparison_data = data[comparisons[0]] # This line is not used in the loop and can be removed
    y_labels = comparisons
    y_positions = np.arange(len(comparisons))
    bar_height = 0.8

    for i, comparison_name in enumerate(comparisons):
        comp_data = data[comparison_name][lang]
        q1_val = comp_data["q1"]
        tie_val = comp_data["tie"]
        q2_val = comp_data["q2"]
        total_count = q1_val + tie_val + q2_val

        # Calculate scaled widths based on fixed_plot_width
        scaled_q1 = (q1_val / total_count) * fixed_plot_width if total_count > 0 else 0
        scaled_tie = (tie_val / total_count) * fixed_plot_width if total_count > 0 else 0
        scaled_q2 = (q2_val / total_count) * fixed_plot_width if total_count > 0 else 0

        parts = [scaled_q1, scaled_tie, scaled_q2] # Use scaled widths
        x_start = 0

        model_names = comparison_name.split(" vs ")
        model_a_name = model_names[0]
        model_b_name = model_names[1]
        label_names = [model_a_name, "Tie", model_b_name]
        original_values = [q1_val, tie_val, q2_val] # Keep original values for percentage calculation

        for j, value in enumerate(parts):
            percentage_value = (original_values[j] / total_count) * 100 if total_count > 0 else 0
            percentage_label = f"{percentage_value:.1f}%" # Format percentage to 1 decimal place

            ax.barh(y=i, width=value, left=x_start, height=bar_height, color=bar_colors[j])
            label_x = x_start + value / 2
            label_y = i
            ax.text(label_x, label_y, f"{label_names[j]}\n{percentage_label}", ha='center', va='center', color='black', fontsize=10, weight='bold') # Use percentage label and bold text
            x_start += value

    ax.set_yticks([]) # Remove y-axis ticks
    ax.set_yticklabels([]) # Remove y-axis labels
    ax.set_xticks([]) # Remove x-axis ticks
    ax.set_xlabel(None) # Remove x-axis label
    ax.set_ylabel(None) # Remove y-axis label
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlim(0, fixed_plot_width) # Set fixed x-axis limit to ensure consistent width
    # ax.set_title(f'Model Comparisons - {lang}', fontsize=30, weight='bold')


    # Save each subplot to a separate PDF file
    filename = f'sbs-{lang.lower()}.pdf'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    # plt.show()
    plt.close(fig) # Close the figure to release memory


# plt.suptitle('Side-by-Side Model Comparison Across Languages', fontsize=16, y=0.99) # Suptitle is not needed for separate plots
plt.show() # Keep show for displaying all in notebook if needed, but not necessary for saving

print("All plots saved.")
